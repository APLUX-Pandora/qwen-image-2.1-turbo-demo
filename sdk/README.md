# Qwen-Image-2.1-Turbo SDK

Load a local model and generate an RGB image from a text prompt.

## What's included

| Path | Content |
| --- | --- |
| `python/` | Python package (`from pandora_qwen_image_2_1_turbo import ImageGenerationPipeline`) |
| `include/` | C header |
| `lib/` | Prebuilt native library with the GGML runtime and Hexagon v73 and v75 DSP libraries |
| `SHA256SUMS` | Checksums of the shipped files |

## Install

In your application's activated Python 3.10–3.12 environment, from the Demo root:

```bash
python -m pip install ./sdk
```

Check the [device requirements](../docs/MODEL_SETUP.md#device-requirements), then [download the upstream models](../README.md#2-get-the-model). The SDK loads the model files from disk.

## Usage

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

## Documentation

- [Python API](PYTHON_API.md)
- [C ABI](../docs/C_ABI.md)
- [Examples](../examples/README.md)
- [Licensing](../LICENSING.md)
