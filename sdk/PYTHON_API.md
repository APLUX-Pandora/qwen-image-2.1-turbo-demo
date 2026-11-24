# Python API

`pandora_qwen_image_2_1_turbo` · Generate RGB images from text with a local Qwen-Image-2.1-Turbo model.

## Quick start

```python
from pandora_qwen_image_2_1_turbo import ImageGenerationPipeline

prompt = (
    "A small orange cat sitting on a wooden windowsill beside a white ceramic "
    "cup and a leafy green plant. Warm afternoon sunlight streams through the "
    "window, casting soft shadows across the wood. Outside, a quiet garden is "
    "gently blurred in the background. Watercolor illustration with delicate "
    "brushwork, subtle paper texture, warm natural colors, and a peaceful "
    "atmosphere."
)
with ImageGenerationPipeline.from_pretrained("models") as pipe:
    result = pipe(prompt, width=512, height=512, seed=42)
result.images[0].save("out.png")
print(result.seed, result.width, result.height)
```

## API overview

| Name | Signature | Returns | Purpose |
| --- | --- | --- | --- |
| `ImageGenerationPipeline.from_pretrained` | `(model_dir, *, runtime=None, n_threads=4, verbose=False)` | `ImageGenerationPipeline` | Load a local model with the bundled runtime. |
| `pipe(...)` | `(prompt, *, width=None, height=None, seed=None, num_inference_steps=None, guidance_scale=None, negative_prompt=None, on_progress=None)` | `ImageGenerationResult` | Generate one image. |
| `pipe.cancel` | `()` | `None` | Request cancellation of active generation. |
| `pipe.close` | `()` | `None` | Release the model. Repeated calls are safe. |
| `pipe.count_tokens` | `(prompt)` | `int` | Count native prompt tokens. An empty prompt is allowed. |
| `pipe.info` | `()` | `dict` | Read the SDK version, supported resolutions, DSP architecture and prompt limits. |
| `pipe.stats` | `()` | `dict` | Read the last generation's encode, denoise, decode and total times in seconds. |
| `pipe.capabilities` | read-only | `ImageGenerationCapabilities` | Default width and height, size multiple, pixel budget and cancellation support. |
| `pipe.needs_reload` | read-only | `bool` | Indicates whether a backend failure requires closing and reloading the model. |
| `ImageGenerationPipeline.image_size` | `(resolution, aspect)` | `tuple[int, int]` | Calculate a width and height preset. Check `info()` for available resolutions. |
| `resolve_models` | `(model_dir=None, *, default_root=None)` | `str` | Resolve an explicit manifest directory, or find it at the supplied default root or its sole manifest-bearing child. |
| `ASPECTS` | constant | `tuple` | Aspect-ratio choices from 1:1 through 9:16. |

## Parameters

### `from_pretrained`

| Parameter | Type | Default | Range / unit | Meaning |
| --- | --- | --- | --- | --- |
| `model_dir` | `str` or path | required | Local directory | Directory containing the [model files and manifest](../docs/MODEL_SETUP.md#folder-layout). Relative paths resolve from the current working directory. |
| `runtime` | `None` | `None` | — | Use the bundled GGML Hexagon runtime. Other values raise `UnsupportedFeatureError`. |
| `n_threads` | `int` | `4` | 1 to 2³¹ − 1 | Host worker threads. |
| `verbose` | `bool` | `False` | — | Enable native diagnostics. |

### `pipe(...)`

| Parameter | Type | Default | Range / unit | Meaning |
| --- | --- | --- | --- | --- |
| `prompt` | `str` | required | UTF-8 text | One nonempty prompt without NUL characters. |
| `width`, `height` | `int` | 1024 each | Pixels, positive multiples of 32 | Set both together. Their product must be at most 2,416,640. |
| `seed` | `int` | Random | 0 to 2⁶³ − 1 | Seed reported in the result. |

Model-specific options

| Parameter | Type | Default | Accepted values | Meaning |
| --- | --- | --- | --- | --- |
| `num_inference_steps` | `int` | 8 | `None` or `8` | Number of denoising steps. The Turbo schedule is fixed at 8. |
| `guidance_scale` | `int` or `float` | 1.0 | `None` or `1.0` | Classifier-free guidance is fixed at 1.0. |
| `negative_prompt` | `str` | `None` | `None` or an empty string | This backend cannot apply a negative prompt. |

Validly typed values outside the supported settings raise `UnsupportedFeatureError`. Pass `on_progress=None` or a callable that receives one `ImageGenerationProgress` argument synchronously. The callback return value is ignored.

Prompt budgets by pixel count are 4,096 tokens up to 1,089,536 pixels and 512 tokens above that. Invalid values raise errors.

## Results and cleanup

### `ImageGenerationResult`

| Field | Type | Meaning |
| --- | --- | --- |
| `images` | List of Pillow images | One owned RGB image, valid after subsequent requests and after close. |
| `seed` | `int` | Seed actually used. |
| `width`, `height` | `int` | Image dimensions in pixels. |

`ImageGenerationProgress` has `stage` (encode, denoise or decode), `completed`, `total` (possibly `None`) and `elapsed_s`. Counts are for the current stage.

### Errors

| Exception | When |
| --- | --- |
| `InputError` | Invalid input, dimensions, seed, token budget or model path. |
| `UnsupportedFeatureError` | The backend cannot honor a requested setting. |
| `ClosedError` | The pipeline is closing or closed. |
| `GenerationCancelledError` | The backend confirmed cancellation. |
| `InferenceError` | Native loading or inference failed. Check `needs_reload` before retrying. |
| `StateError` | A callback attempted a disallowed operation on its active pipeline. |
| `TypeError` | Invalid signature or unknown keyword. |

SDK exceptions derive from `QwiError`, which exposes the native status as `code` where available.

### Lifecycle

Use a context manager or call `close()` in a `finally` block. One pipeline may be live per process. A pipeline serializes generation and state operations. `count_tokens()` and `cancel()` can run during generation or inside its callback. Keep callbacks short. A callback exception requests cancellation and propagates after native execution returns.

`cancel()` requests a stop at a safe checkpoint. If generation finishes first, the caller receives the completed image. `close()` rejects new calls, cancels active generation and waits for native callers before releasing resources. Call it outside callbacks and signal handlers.

## Advanced

See the [examples](../examples/README.md) for preset sizes, batch processing and coordinated signal handling.
