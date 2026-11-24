#!/usr/bin/env python3
"""Generate one image per line of a prompt file with one engine, e.g. a set of variations for review.

The engine is opened once; every line becomes an image named <out-dir>/<n>_<seed>.png. Ctrl+C stops the running image
(coordinated pipeline.cancel) and the loop. Lines that are empty or start with # are skipped.

usage: python3 examples/python/batch_prompts.py prompts.txt --out-dir out/batch [--tier 1024] [--aspect 1:1]
       [--seed 42] [--seeds 3] [--models models]
"""
import argparse
import os
import signal
import sys
import time
import threading

# Signed SDK files stay immutable during normal example use.
sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandora_qwen_image_2_1_turbo import resolve_models, ASPECTS, ImageGenerationPipeline, GenerationCancelledError, QwiError  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompts", nargs="?", help="text file, one prompt per line")
    ap.add_argument("--out-dir", default="out/batch")
    ap.add_argument("--tier", type=int, choices=[512, 1024, 1536, 2048])
    ap.add_argument("--aspect", default="1:1", choices=ASPECTS)
    ap.add_argument("--seed", type=int, default=42, help="first seed")
    ap.add_argument("--seeds", type=int, default=1, help="images per prompt (seed, seed + 1, ...)")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()

    if a.tier is None:
        a.tier = 512 if a.prompts is None else 1024
    if a.prompts is None:
        a.prompts = os.path.join(ROOT, "examples", "prompts.txt")

    if a.seeds <= 0 or a.seed < 0 or a.seed + a.seeds - 1 >= 2**63:
        ap.error("seeds must be positive and all seeds must fit nonnegative int64")
    prompts = [ln.strip() for ln in open(a.prompts, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    if any("\0" in prompt for prompt in prompts):
        ap.error("prompts must not contain NUL characters")
    w, h = ImageGenerationPipeline.image_size(a.tier, a.aspect)
    os.makedirs(a.out_dir, exist_ok=True)
    signals = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    stop, receiver_done = threading.Event(), threading.Event()
    owner = []

    def receive_stop():
        while not receiver_done.is_set():
            if signal.sigtimedwait(signals, 0.2) is not None:
                stop.set()
                if owner:
                    owner[0].cancel()

    receiver = threading.Thread(target=receive_stop, name="batch-stop-signals")
    receiver.start()
    t_all, n = time.time(), 0
    try:
        model_root = resolve_models(a.models, default_root=os.path.join(ROOT, "models"))
        print(f"[models: {model_root}]", file=sys.stderr)
        with ImageGenerationPipeline.from_pretrained(model_root) as eng:
            owner.append(eng)

            def progress(*_):
                if stop.is_set():
                    eng.cancel()

            for i, prompt in enumerate(prompts):
                for k in range(a.seeds):
                    if stop.is_set():
                        break
                    seed = a.seed + k
                    path = os.path.join(a.out_dir, f"{i:02d}_{seed}.png")
                    try:
                        result = eng(prompt, width=w, height=h, seed=seed, on_progress=progress)
                    except GenerationCancelledError:
                        break
                    except QwiError as e:
                        print(f"[{path}: failed: {e}]", file=sys.stderr)
                        continue
                    if stop.is_set():
                        break
                    result.images[0].save(path, format="PNG")
                    n += 1
                    print(f"[{path}: {eng.stats()['total_s']:.1f} s] {prompt[:70]}", file=sys.stderr)
    finally:
        receiver_done.set()
        receiver.join()
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    print(f"[{n} images of {w} x {h} in {time.time() - t_all:.1f} s{' (stopped)' if stop.is_set() else ''}]", file=sys.stderr)


if __name__ == "__main__":
    main()
