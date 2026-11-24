"""Qwen local image-generation pipeline and internal C ABI adapter.

Use ImageGenerationPipeline.from_pretrained(), pipe(prompt, ...), and
result.images[0].save(). See sdk/PYTHON_API.md for the supported contract.
The native adapter implements the pipeline's lifecycle.
"""
import ctypes
import os
import signal
import threading
import math
import secrets
from typing import NamedTuple, Optional
from contextlib import contextmanager

__version__ = "1.0.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDK = os.path.dirname(_HERE)

ASPECTS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16")
PHASES = ("encode", "denoise", "decode")

OK, ERR_ARG, ERR_MODEL, ERR_NPU, ERR_STATE, ERR_CANCEL, ERR_UNSUPPORTED, ERR_INTERNAL = 0, -1, -2, -3, -4, -5, -6, -7


class QwiError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _Options(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int), ("verbose", ctypes.c_int)]


class _Request(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p), ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("seed", ctypes.c_int64)]


class _Info(ctypes.Structure):
    _fields_ = [("version", ctypes.c_char_p), ("n_steps", ctypes.c_int), ("n_resolutions", ctypes.c_int),
                ("resolutions", ctypes.c_int * 4), ("hexagon_arch", ctypes.c_int), ("max_prompt_tokens", ctypes.c_int * 4)]


class _Stats(ctypes.Structure):
    _fields_ = [("encode_ms", ctypes.c_double), ("denoise_ms", ctypes.c_double), ("decode_ms", ctypes.c_double),
                ("total_ms", ctypes.c_double)]


_PROGRESS = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_void_p)


def _find_lib():
    env = os.environ.get("QWEN_IMAGE_LIB")
    if env:
        return env, False
    try:
        from _pandora_qwen_image_2_1_turbo_runtime import library_path
    except ImportError:
        pass
    else:
        return str(library_path()), False
    for d, from_build in ((f"{_SDK}/lib", False), (f"{_SDK}/build", True)):
        if os.path.exists(f"{d}/libqwen_image.so"):
            return f"{d}/libqwen_image.so", from_build
    raise FileNotFoundError("libqwen_image.so not found in sdk/lib or sdk/build (set QWEN_IMAGE_LIB)")


def _load():
    path, from_build = _find_lib()
    dsp = f"{_SDK}/dsp/out/lib"
    if from_build and "ADSP_LIBRARY_PATH" not in os.environ and os.path.isdir(dsp):
        os.environ["ADSP_LIBRARY_PATH"] = f"{dsp};/usr/lib/rfsa/adsp;/dsp"
    lib = ctypes.CDLL(path)
    lib.qwi_default_options.restype = _Options
    lib.qwi_create.restype = ctypes.c_void_p
    lib.qwi_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Options)]
    lib.qwi_destroy.argtypes = [ctypes.c_void_p]
    lib.qwi_image_size.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.qwi_generate.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Request), _PROGRESS, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_size_t]
    lib.qwi_cancel.argtypes = [ctypes.c_void_p]
    lib.qwi_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Info)]
    lib.qwi_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
    lib.qwi_count_tokens.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.qwi_last_error.restype = ctypes.c_char_p
    lib.qwi_last_error.argtypes = [ctypes.c_void_p]
    lib.qwi_create_error.restype = ctypes.c_char_p
    return lib


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


@contextmanager
def _defer_stop_signals():
    """Keep ownership transitions atomic with respect to POSIX stop signals."""
    mask = None
    if hasattr(signal, "pthread_sigmask"):
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        yield
    finally:
        if mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def resolve_models(model_dir=None, *, default_root=None):
    """Resolve an explicit manifest root, or a unique package under this project's models/."""
    if model_dir is not None:
        root = os.path.abspath(os.fspath(model_dir))
        if not os.path.isfile(os.path.join(root, "models.json")):
            raise QwiError(ERR_MODEL, f"model directory must contain models.json: {root}")
        return root
    if default_root is None:
        if os.path.basename(_HERE) != "python" or os.path.basename(_SDK) != "sdk":
            raise QwiError(ERR_MODEL, "pass an explicit model root when using the installed SDK")
        default_root = os.path.join(os.path.dirname(_SDK), "models")
    root = os.path.abspath(os.fspath(default_root))
    if os.path.isfile(os.path.join(root, "models.json")):
        return root
    candidates = sorted(entry.path for entry in os.scandir(root)
                        if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "models.json"))) if os.path.isdir(root) else []
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise QwiError(ERR_MODEL, "multiple model packages found; pass an explicit manifest root: " + ", ".join(candidates))
    raise QwiError(ERR_MODEL, f"no model manifest found at {root} or its immediate children; "
                   "follow docs/MODEL_SETUP.md or pass an explicit model directory")


class _Engine:
    def __init__(self, model_dir=None, n_threads=4, verbose=False):
        self._e = None
        self._state = threading.Condition(threading.RLock())
        self._active_thread = None
        self._token_readers = 0
        self._closing = False
        self.model_dir = resolve_models(model_dir)
        lib = _get_lib()
        opt = lib.qwi_default_options()
        opt.n_threads, opt.verbose = n_threads, int(bool(verbose))
        try:
            with _defer_stop_signals():
                self._e = lib.qwi_create(os.fsencode(self.model_dir), ctypes.byref(opt))
                if not self._e:
                    raise QwiError(ERR_MODEL, lib.qwi_create_error().decode("utf-8", "replace"))
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _operation(self):
        with _defer_stop_signals():
            ident = threading.get_ident()
            with self._state:
                if self._active_thread == ident:
                    raise QwiError(ERR_STATE, "only cancel() and count_tokens() may be called from a progress callback")
                while self._active_thread is not None and not self._closing:
                    self._state.wait()
                if self._closing or not self._e:
                    raise QwiError(ERR_STATE, "engine is closed or closing")
                self._active_thread = ident
                handle = self._e
            try:
                yield handle
            finally:
                with self._state:
                    self._active_thread = None
                    self._state.notify_all()

    @staticmethod
    def image_size(resolution, aspect):
        """Return a model-card aspect-ratio preset for a resolution tier."""
        w, h = ctypes.c_int(), ctypes.c_int()
        if _get_lib().qwi_image_size(int(resolution), ASPECTS.index(aspect), ctypes.byref(w), ctypes.byref(h)) != OK:
            raise QwiError(ERR_ARG, f"unknown resolution tier {resolution}")
        return w.value, h.value

    def info(self):
        with self._operation() as handle:
            i = _Info()
            _get_lib().qwi_get_info(handle, ctypes.byref(i))
            return {"version": i.version.decode(), "n_steps": i.n_steps,
                    "resolutions": list(i.resolutions[:i.n_resolutions]), "hexagon_arch": i.hexagon_arch,
                    "max_prompt_tokens": dict(zip(i.resolutions[:i.n_resolutions], i.max_prompt_tokens[:i.n_resolutions]))}

    def count_tokens(self, prompt):
        """Count tokens using the model's native tokenizer; safe during generation."""
        with _defer_stop_signals():
            with self._state:
                if self._closing or not self._e:
                    raise QwiError(ERR_STATE, "engine is closed or closing")
                self._token_readers += 1
                handle = self._e
            try:
                n = _get_lib().qwi_count_tokens(handle, prompt.encode("utf-8"))
                if n < 0:
                    raise QwiError(n, "cannot count the tokens of this prompt")
                return n
            finally:
                with self._state:
                    self._token_readers -= 1
                    self._state.notify_all()

    def stats(self):
        with self._operation() as handle:
            s = _Stats()
            _get_lib().qwi_get_stats(handle, ctypes.byref(s))
            return {"encode_s": s.encode_ms / 1000, "denoise_s": s.denoise_ms / 1000, "decode_s": s.decode_ms / 1000,
                    "total_s": s.total_ms / 1000}

    def _generate_rgb(self, prompt, width, height, seed, progress=None):
        """Generate RGB8 bytes. Only cancel() and count_tokens() are permitted inside progress callbacks."""
        lib = _get_lib()
        out = ctypes.create_string_buffer(width * height * 3)
        req = _Request(prompt.encode("utf-8"), width, height, seed)
        callback_errors = []

        def cb(phase, step, n, t, _user):
            if progress is not None:
                try:
                    progress(PHASES[phase], step, n, t)
                except BaseException as error:
                    if not callback_errors:
                        callback_errors.append(error)
                    self.cancel()

        c_cb = _PROGRESS(cb)
        with self._operation() as handle:
            rc = lib.qwi_generate(handle, ctypes.byref(req), c_cb, None, out, len(out))
            if callback_errors:
                raise callback_errors[0]
            if rc != OK:
                raise QwiError(rc, lib.qwi_last_error(handle).decode("utf-8", "replace"))
            return out.raw

    def cancel(self):
        with self._state:
            if self._e:
                with _defer_stop_signals():
                    _get_lib().qwi_cancel(self._e)

    def close(self):
        """Cancel and join active native calls before destroying the handle; idempotent."""
        with _defer_stop_signals():
            with self._state:
                if self._active_thread == threading.get_ident():
                    raise QwiError(ERR_STATE, "close() cannot run inside a progress callback")
                while self._closing:
                    self._state.wait()
                if not self._e:
                    return
                self._closing = True
                try:
                    _get_lib().qwi_cancel(self._e)
                    while self._active_thread is not None or self._token_readers:
                        self._state.wait()
                    _get_lib().qwi_destroy(self._e)
                    self._e = None
                finally:
                    self._closing = False
                    self._state.notify_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class InputError(QwiError):
    """Invalid input or local model resources."""
    def __init__(self, message, code=ERR_ARG):
        super().__init__(code, message)


class UnsupportedFeatureError(QwiError):
    def __init__(self, message, code=ERR_UNSUPPORTED):
        super().__init__(code, message)


class ClosedError(QwiError):
    def __init__(self, message, code=ERR_STATE):
        super().__init__(code, message)


class GenerationCancelledError(QwiError):
    def __init__(self, message, code=ERR_CANCEL):
        super().__init__(code, message)


class InferenceError(QwiError):
    def __init__(self, message, code=ERR_INTERNAL):
        super().__init__(code, message)


class StateError(QwiError):
    """An operation would reenter the active callback's own request."""
    def __init__(self, message, code=ERR_STATE):
        super().__init__(code, message)


def _classified(error):
    kind = {ERR_ARG: InputError, ERR_MODEL: InferenceError, ERR_CANCEL: GenerationCancelledError,
            ERR_UNSUPPORTED: UnsupportedFeatureError, ERR_STATE: StateError}.get(error.code, InferenceError)
    return kind(str(error), code=error.code)


# Named tuples keep metadata read-only without a dependency on a shared SDK package.
class ImageGenerationResult(NamedTuple):
    images: list
    seed: int
    width: int
    height: int


class ImageGenerationProgress(NamedTuple):
    stage: str
    completed: int
    total: Optional[int]
    elapsed_s: float


class ImageGenerationCapabilities(NamedTuple):
    default_width: int = 1024
    default_height: int = 1024
    size_multiple: int = 32
    max_pixels: int = 2416640
    supports_cancel: bool = True


class ImageGenerationPipeline(_Engine):
    """Local text-to-image pipeline over the existing Qwen C ABI.

    The native implementation supports one live instance per process. Separate
    processes are not coordinated here; the runtime owns hardware scheduling.
    """
    def __init__(self, model_dir=None, *, n_threads=4, verbose=False):
        if type(n_threads) is not int or not 1 <= n_threads <= 2**31 - 1:
            raise InputError("n_threads must be a positive C-int")
        if type(verbose) is not bool:
            raise InputError("verbose must be a boolean")
        self._failed = False
        self._capabilities = ImageGenerationCapabilities()
        try:
            model_dir = resolve_models(model_dir)
        except (QwiError, OSError, TypeError, ValueError) as error:
            raise InputError(str(error), code=ERR_MODEL) from error
        try:
            super().__init__(model_dir, n_threads=n_threads, verbose=verbose)
        except QwiError as error:
            raise InferenceError(str(error), code=error.code) from error
        except (OSError, TypeError, ValueError) as error:
            raise InferenceError("Cannot load Qwen SDK: " + str(error)) from error

    @classmethod
    def from_pretrained(cls, model_dir, *, runtime=None, n_threads=4, verbose=False):
        """Load a local model using the bundled GGML runtime."""
        if runtime is not None:
            raise UnsupportedFeatureError("This pipeline uses its bundled GGML runtime; set runtime to None")
        return cls(model_dir, n_threads=n_threads, verbose=verbose)

    @property
    def needs_reload(self):
        with self._state:
            return self._failed

    @property
    def capabilities(self):
        return self._capabilities

    @contextmanager
    def _operation(self):
        try:
            with super()._operation() as handle:
                if self._failed:
                    raise InferenceError("A native failure requires close and reload")
                try:
                    yield handle
                except QwiError as error:
                    if type(error) is QwiError and error.code in (ERR_MODEL, ERR_NPU, ERR_INTERNAL):
                        self._failed = True
                    raise
        except QwiError as error:
            if type(error) is not QwiError:
                raise
            if error.code == ERR_STATE:
                kind = ClosedError if self._closing or not self._e else StateError
                raise kind(str(error)) from error
            raise _classified(error) from error

    def count_tokens(self, prompt):
        self._validate_prompt(prompt, require_nonempty=False)
        try:
            return super().count_tokens(prompt)
        except QwiError as error:
            if error.code == ERR_STATE and (self._closing or not self._e):
                raise ClosedError(str(error)) from error
            raise _classified(error) from error

    @staticmethod
    def _validate_prompt(prompt, require_nonempty=True):
        if not isinstance(prompt, str) or "\0" in prompt or (require_nonempty and not prompt.strip()):
            raise InputError("prompt must be a nonempty NUL-free string")
        try:
            prompt.encode("utf-8")
        except UnicodeError as error:
            raise InputError("prompt must be valid UTF-8 text") from error

    def __call__(self, prompt, *, width=None, height=None, seed=None,
                 num_inference_steps=None, guidance_scale=None, negative_prompt=None,
                 on_progress=None):
        # Reject callback recursion before doing any validation or token work.
        with self._state:
            if self._active_thread == threading.get_ident():
                raise StateError("generation cannot reenter a progress callback")
            if self._closing or not self._e:
                raise ClosedError("pipeline is closed or closing")
            if self._failed:
                raise InferenceError("A native failure requires close and reload")
        self._validate_prompt(prompt)
        if width is None and height is None:
            width = height = 1024
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise InputError("width and height must be paired positive integers")
        if width % 32 or height % 32 or width * height > self.capabilities.max_pixels:
            raise InputError("dimensions must be multiples of 32 within the 2416640-pixel budget")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
            raise InputError("seed must be None or an integer in [0, 2**63-1]")
        if num_inference_steps is not None:
            if type(num_inference_steps) is not int or num_inference_steps <= 0:
                raise InputError("num_inference_steps must be a positive integer")
            if num_inference_steps != 8:
                raise UnsupportedFeatureError("This Turbo backend uses a fixed 8-step schedule")
        if guidance_scale is not None:
            if type(guidance_scale) not in (int, float):
                raise InputError("guidance_scale must be a finite real number")
            try:
                finite = math.isfinite(guidance_scale)
            except OverflowError:
                finite = False
            if not finite:
                raise InputError("guidance_scale must be a finite real number")
            if guidance_scale != 1:
                raise UnsupportedFeatureError("This Turbo backend uses fixed guidance_scale=1")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise InputError("negative_prompt must be None or a string")
        if negative_prompt:
            raise UnsupportedFeatureError("This backend does not support negative_prompt")
        if on_progress is not None and not callable(on_progress):
            raise InputError("on_progress must be callable or None")
        pixels = width * height
        limit = next(tokens for budget, tokens in ((272384, 4096), (1089536, 4096), (2416640, 512)) if pixels <= budget)
        tokens = self.count_tokens(prompt)
        if tokens > limit:
            raise InputError(f"prompt has {tokens} tokens; at most {limit} at this image size")
        actual_seed = secrets.randbits(63) if seed is None else seed
        # Import before generation so a missing image dependency never wastes an
        # inference. frombytes copies the returned RGB bytes into an owned image.
        from PIL import Image

        def progress(phase, step, total, elapsed):
            with self._state:
                closing = self._closing
            # Native entry resets cancellation; reapply a close that raced entry.
            if closing:
                self.cancel()
            if on_progress is not None:
                on_progress(ImageGenerationProgress(phase, step, total if total > 0 else None, elapsed))

        rgb = super()._generate_rgb(prompt, width, height, actual_seed, progress)
        image = Image.frombytes("RGB", (width, height), rgb)
        return ImageGenerationResult([image], actual_seed, width, height)

    def close(self):
        try:
            return super().close()
        except QwiError as error:
            raise _classified(error) from error

    def __enter__(self):
        with self._state:
            if self._closing or not self._e:
                raise ClosedError("pipeline is closed or closing")
        return self
