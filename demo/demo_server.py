#!/usr/bin/env python3
"""Qwen-Image-2.1-Turbo web demo server (runs on the board).

One page (static/index.html): prompt, resolution tier, aspect ratio and seed in, one image out. Requests go into a
queue and run one at a time through this service owner; progress (queue position, prompt encoding, each of the 8
denoising steps, decoding) is pushed to every open page over Server-Sent Events.

Engine: the SDK (sdk/python/pandora_qwen_image_2_1_turbo.py over libqwen_image.so), created once at start-up; progress comes from its
callback. Images are written to out/web/<id>.png.

The SDK pipeline uses Pillow for images.
usage: demo/run.sh [--port 8810]   (env: QWEN_IMAGE_MODELS, QWEN_IMAGE_LIB; see demo/README.md)
"""
import argparse
import collections
import itertools
import json
import os
import queue
import re
import signal
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = f"{HERE}/static"
OUT = f"{ROOT}/out/web"
MODELS = os.environ.get("QWEN_IMAGE_MODELS")

from pandora_qwen_image_2_1_turbo import resolve_models, ImageGenerationPipeline, GenerationCancelledError, QwiError  # noqa: E402

N_STEPS = 8
MAX_SEED = 2**31 - 1
MAX_PROMPT = 40000     # characters: only a guard against oversized requests; the real limit is in tokens
QUEUE_LIMIT = 8
MAX_BODY = 256 * 1024
JOB_HISTORY_LIMIT = 64
# The model card's aspect-ratio presets at the 2048 base; a tier scales them and rounds to multiples of 32.
BASE_RESOLUTION = 2048
ASPECT_RATIOS = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
    "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752),
}
# The model card's tiers (the official demo has 1K / 1.5K / 2K) plus 512 for quick pictures of simple subjects.
ALL_TIERS = [512, 1024, 1536, 2048]
# Measured on the board at 1:1 (s): prompt encoding, one denoising step, decoding. Used for the page's estimates.
ESTIMATES = {512: (7.3, 4.1, 2.9), 1024: (5.2, 17.3, 17.5), 1536: (6.3, 50.1, 45.5)}


def image_size(tier, ratio):
    w, h = ASPECT_RATIOS[ratio]
    s = tier / BASE_RESOLUTION
    return max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)


class Job:
    _ids = itertools.count(1)

    def __init__(self, prompt, tier, ratio, seed):
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{next(self._ids)}"
        self.prompt, self.tier, self.ratio, self.seed = prompt, tier, ratio, seed
        self.width, self.height = image_size(tier, ratio)
        self.cancelled = False


class SdkEngine:
    """libqwen_image.so, created once; one generate() at a time (the worker thread)."""

    def __init__(self):
        t0 = time.time()
        model_root = resolve_models(MODELS, default_root=Path(ROOT) / "models")
        print(json.dumps({"model_root": model_root}), flush=True)
        self.eng = ImageGenerationPipeline.from_pretrained(model_root)
        try:
            self.info = self.eng.info()
            self.init_s = time.time() - t0
        except BaseException:
            self.eng.close()
            raise

    def run(self, job, emit):
        if job.cancelled:
            return None, "cancelled"
        os.makedirs(OUT, exist_ok=True)

        def progress(event):
            phase, step, n, _t = event
            # Native entry resets cancellation for a new inference. Reapply any
            # cancellation that arrived after the worker published this job.
            if job.cancelled:
                self.eng.cancel()
                return
            if phase == "encode":
                emit({"type": "phase", "phase": "encode"})
            elif phase == "denoise":
                emit({"type": "phase", "phase": "denoise", "step": 0, "n_steps": n} if step == 0
                     else {"type": "step", "step": step, "n_steps": n})
            elif phase == "decode":
                emit({"type": "phase", "phase": "decode"} if step == 0
                     else {"type": "decode_step", "step": step, "n_steps": n})

        try:
            result = self.eng(job.prompt, width=job.width, height=job.height, seed=job.seed, on_progress=progress)
        except GenerationCancelledError:
            return None, "cancelled"
        except QwiError as e:
            return None, str(e) or f"Native inference failed (code {e.code})"
        if job.cancelled:
            return None, "cancelled"
        result.images[0].save(f"{OUT}/{job.id}.png", format="PNG")
        return {k: round(v, 2) for k, v in self.eng.stats().items()}, None

    def cancel(self, job):
        job.cancelled = True
        self.eng.cancel()


class Server:
    def __init__(self):
        self.stopping = threading.Event()
        self.engine = SdkEngine()
        self.tiers = [t for t in self.engine.info["resolutions"] if t in ALL_TIERS]
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.model_info = self.engine.info
        self.loading = False
        self.failure = None
        self.pending = collections.deque()
        self.running = None
        self.wake = threading.Event()
        self.subs = []
        self.events_lock = threading.Lock()
        self.job_events = collections.OrderedDict()
        self.worker_thread = threading.Thread(target=self.worker, name="image-inference")
        try:
            self.worker_thread.start()
        except BaseException:
            self.engine.eng.close()
            raise

    def info(self):
        return {"type": "info", "tiers": self.tiers, "all_tiers": ALL_TIERS, "n_steps": N_STEPS,
                "aspect_ratios": list(ASPECT_RATIOS), "max_seed": MAX_SEED,
                "sizes": {t: {r: image_size(t, r) for r in ASPECT_RATIOS} for t in ALL_TIERS},
                "estimates": {t: {"encode_s": e, "step_s": s, "decode_s": d} for t, (e, s, d) in ESTIMATES.items()},
                "max_tokens": {t: self.model_info["max_prompt_tokens"][t] for t in self.tiers},
                "ready": not self.stopping.is_set() and not self.loading and self.failure is None,
                "reload_required": self.failure is not None, "loading": self.loading,
                "model_error": self.failure,
                "busy": self.running is not None, "queued": len(self.pending)}

    def broadcast(self, ev):
        with self.events_lock:
            if "id" in ev:
                self.job_events[ev["id"]] = dict(ev)
                self.job_events.move_to_end(ev["id"])
                while len(self.job_events) > JOB_HISTORY_LIMIT:
                    self.job_events.popitem(last=False)
            for q in list(self.subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    # Disconnect a stalled subscriber; job status remains queryable.
                    self.subs.remove(q)
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait(None)

    def job_status(self, job_id):
        with self.events_lock:
            ev = self.job_events.get(job_id)
            if ev is not None:
                return dict(ev)
        with self.lock:
            if any(j.id == job_id for j in self.pending) or (self.running and self.running.id == job_id):
                return {"type": "pending", "id": job_id}
        # A terminal event may have arrived while the queue lock was acquired.
        with self.events_lock:
            return dict(self.job_events[job_id]) if job_id in self.job_events else None

    def submit(self, b):
        if self.stopping.is_set():
            return None, "service is stopping"
        prompt = str(b.get("prompt", "")).strip()
        try:
            tier, seed = int(b.get("tier", 512)), int(b.get("seed", 0))
        except (TypeError, ValueError, OverflowError):
            return None, "tier and seed must be integers"
        ratio = str(b.get("ratio", "1:1"))
        if not prompt:
            return None, "the prompt is empty"
        if len(prompt) > MAX_PROMPT:
            return None, f"the prompt is longer than {MAX_PROMPT} characters"
        if ratio not in ASPECT_RATIOS:
            return None, f"unknown aspect ratio {ratio}"
        if tier not in self.tiers:
            return None, f"resolution {tier} is not available on this device"
        with self.lock:
            if self.loading or self.failure is not None or self.engine is None:
                return None, "model unavailable; wait for loading or reload explicitly"
            owner = self.engine
        n, cap = owner.eng.count_tokens(prompt), self.model_info["max_prompt_tokens"][tier]
        if n > cap:
            return None, f"the prompt has {n} tokens; at most {cap} at this resolution"
        if not 0 <= seed <= MAX_SEED:
            return None, f"the seed must be between 0 and {MAX_SEED}"
        with self.lock:
            if self.stopping.is_set():
                return None, "service is stopping"
            if self.loading or self.failure is not None:
                return None, "model unavailable; reload explicitly"
            if len(self.pending) >= QUEUE_LIMIT:
                return None, "too many requests are waiting; try again later"
            job = Job(prompt, tier, ratio, seed)
            self.pending.append(job)
            pos = len(self.pending) + (self.running is not None) - 1
        self.wake.set()
        self.broadcast_queue()
        return {"id": job.id, "width": job.width, "height": job.height, "ahead": pos}, None

    def count_tokens(self, prompt):
        with self.lock:
            if self.stopping.is_set() or self.loading or self.failure is not None or self.engine is None:
                raise QwiError(-3, "Model unavailable; wait for loading or reload explicitly")
            return self.engine.eng.count_tokens(prompt)

    def cancel(self, job_id):
        with self.lock:
            waiting = next((j for j in self.pending if j.id == job_id), None)
            if waiting:
                self.pending.remove(waiting)
            elif self.running and self.running.id == job_id:
                self.engine.cancel(self.running)
                return True
            else:
                return False
        self.broadcast({"type": "cancelled", "id": job_id})
        self.broadcast_queue()
        return True

    def broadcast_queue(self):
        with self.lock:
            ids = ([self.running.id] if self.running else []) + [j.id for j in self.pending]
        state = self.info()
        self.broadcast({"type": "queue", "ids": ids, **{k: state[k] for k in ("ready", "reload_required", "loading", "model_error")}})

    def worker(self):
        while True:
            self.wake.wait()
            with self.lock:
                if getattr(self, "stopping", None) is not None and self.stopping.is_set():
                    return
                if not self.pending:
                    self.wake.clear()
                    continue
                job = self.running = self.pending.popleft()
            self.broadcast_queue()
            t0 = time.time()
            emit = lambda ev: self.broadcast({**ev, "id": job.id, "elapsed_s": round(time.time() - t0, 1)})  # noqa: E731
            emit({"type": "start", "width": job.width, "height": job.height, "seed": job.seed})
            print(f"[job] {job.id} {job.width}x{job.height} seed {job.seed} prompt {len(job.prompt)} chars", flush=True)
            try:
                times, err = self.engine.run(job, emit)
            except Exception as e:  # report and keep serving
                times, err = None, f"internal error: {e}"
            # Linearize completion with cancel(): an acknowledged cancellation
            # must not turn into a done event while the PNG is being finalized.
            with self.lock:
                if job.cancelled:
                    err = "cancelled"
                if err == "cancelled":
                    emit({"type": "cancelled"})
                elif err:
                    emit({"type": "error", "message": err})
                else:
                    emit({"type": "done", "image": f"/api/image/{job.id}.png", "width": job.width, "height": job.height,
                          "seed": job.seed, "tier": job.tier, "ratio": job.ratio, "prompt": job.prompt,
                          "wall_s": round(time.time() - t0, 1), **times})
                if getattr(getattr(self.engine, "eng", None), "needs_reload", False):
                    self.failure = err or "native execution failed"
                    while self.pending:
                        waiting = self.pending.popleft()
                        self.broadcast({"type": "error", "id": waiting.id, "message": "Model requires explicit reload"})
                self.running = None
            print(f"[job] {job.id} {err or 'done'} {time.time() - t0:.1f}s", flush=True)
            self.broadcast_queue()

    def reload(self):
        """Replace the owner only on explicit request, after all work has stopped."""
        with self.lifecycle_lock:
            with self.lock:
                if self.stopping.is_set(): raise ValueError("service is stopping")
                if self.running is not None or self.pending: raise ValueError("wait for running and queued jobs before reload")
                self.loading = True
                old, self.engine = self.engine, None
            try:
                if old is not None: old.eng.close()
                replacement = SdkEngine()
                with self.lock:
                    self.engine = replacement
                    self.model_info = replacement.info
                    self.failure = None
            except Exception as error:
                with self.lock: self.failure = str(error)
                raise QwiError(-3, "Model reload failed: " + str(error)) from error
            finally:
                with self.lock: self.loading = False
                self.broadcast_queue()
        return self.info()

    def close(self):
        """Stop admission; join inference and reload owners before destruction."""
        with self.lock:
            self.stopping.set()
            for job in self.pending: job.cancelled = True
            self.pending.clear()
            if self.running is not None: self.engine.cancel(self.running)
        self.wake.set()
        self.worker_thread.join()
        with getattr(self, "lifecycle_lock", self.lock):
            if self.engine is not None: self.engine.eng.close()
        print(json.dumps({"engine_closed": True, "inference_thread_joined": True}), flush=True)
        with self.events_lock:
            for q in self.subs:
                try: q.put_nowait(None)
                except queue.Full: pass


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(2)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 <= n <= MAX_BODY:
            raise ValueError("request body is too large or has an invalid length")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError("request body must be valid UTF-8 JSON") from e
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _file(self, path, ctype, cache=False):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._file(f"{STATIC}/index.html", "text/html; charset=utf-8")
        assets = {"/static/logo.png": ("logo.png", "image/png"),
                  "/static/pandora-mark.svg": ("pandora-mark.svg", "image/svg+xml")}
        if p in assets:
            name, content_type = assets[p]
            return self._file(f"{STATIC}/{name}", content_type, cache=True)
        if p == "/api/info":
            return self._json(S.info())
        if p == "/api/events":
            return self._events()
        m = re.fullmatch(r"/api/job/([0-9A-Za-z-]+)", p)
        if m:
            ev = S.job_status(m.group(1))
            return self._json(ev if ev is not None else {"error": "job no longer available"},
                              200 if ev is not None else 404)
        m = re.fullmatch(r"/api/image/([0-9A-Za-z-]+)\.png", p)
        if m:
            return self._file(f"{OUT}/{m.group(1)}.png", "image/png", cache=True)
        self.send_error(404)

    def do_POST(self):
        try:
            return self._post()
        except (ValueError, QwiError) as e:
            return self._json({"error": str(e)}, 400)

    def _post(self):
        p = self.path.split("?")[0]
        b = self._body()
        if p == "/api/reload":
            try: return self._json(S.reload())
            except ValueError as e: return self._json({"error": str(e)}, 409)
            except QwiError as e: return self._json({"error": str(e), "reload_required": True}, 503)
        if p == "/api/generate":
            res, err = S.submit(b)
            return self._json(res if err is None else {"error": err}, 200 if err is None else (503 if not S.info()["ready"] else 400))
        if p == "/api/tokens":
            prompt = str(b.get("prompt", ""))
            if len(prompt) > MAX_PROMPT:
                raise ValueError(f"the prompt is longer than {MAX_PROMPT} characters")
            try: return self._json({"tokens": S.count_tokens(prompt)})
            except QwiError as error: return self._json({"error": str(error)}, 503)
        if p == "/api/cancel":
            return self._json({"cancelled": S.cancel(str(b.get("id", "")))})
        self.send_error(404)

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.connection.settimeout(2)
        q = queue.Queue(maxsize=128)
        with S.events_lock:
            S.subs.append(q)
        try:
            self.wfile.write(f"data: {json.dumps(S.info())}\n\n".encode())
            self.wfile.flush()
            while not S.stopping.is_set():
                try:
                    ev = q.get(timeout=0.5)
                    if ev is None:
                        break
                    self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            with S.events_lock:
                if q in S.subs:
                    S.subs.remove(q)


class DemoHTTPServer(ThreadingHTTPServer):
    """Bind the configured IP without an unnecessary reverse-DNS lookup."""
    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]


def main():
    global S
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8810)
    a = ap.parse_args()
    stops = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, stops)
    stop = threading.Event()
    receiver_done = threading.Event()

    def receive_stop():
        while not receiver_done.is_set():
            received = signal.sigtimedwait(stops, 0.2)
            if received is not None:
                stop.set()

    receiver = threading.Thread(target=receive_stop, name="stop-signals")
    receiver.start()
    S = None
    srv = None
    try:
        S = Server()
        i = S.engine.info
        print(f"[demo] engine {i['version']} ready in {S.engine.init_s:.1f}s (Hexagon v{i['hexagon_arch']}, tiers {i['resolutions']}); "
              f"http://0.0.0.0:{a.port}/  (images in out/web)", flush=True)
        if not stop.is_set():
            srv = DemoHTTPServer(("0.0.0.0", a.port), Handler)
            srv.daemon_threads = False
            srv.timeout = 0.2
            while not stop.is_set():
                srv.handle_request()
    finally:
        try:
            if S is not None:
                S.close()
        finally:
            if srv is not None:
                srv.server_close()
            receiver_done.set()
            receiver.join()
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


if __name__ == "__main__":
    main()
