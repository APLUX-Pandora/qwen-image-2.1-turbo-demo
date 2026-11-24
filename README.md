# Qwen-Image-2.1-Turbo

<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo"><img src="https://img.shields.io/badge/Hugging%20Face-QWEN--IMAGE--2.1--TURBO-E3B800?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=555555" alt="Hugging Face: Qwen-Image-2.1-Turbo"></a>
  <a href="https://github.com/APLUX-Pandora/qwen-image-2.1-turbo-demo"><img src="https://img.shields.io/badge/GitHub-QWEN--IMAGE--2.1--TURBO-000000?style=for-the-badge&amp;logo=github&amp;logoColor=white&amp;labelColor=555555" alt="GitHub: Qwen-Image-2.1-Turbo"></a>
</p>

Generate images from text on a compatible device using the GGML Hexagon backend. This Demo uses FastRPC under a runtime exception covering the public upstream GGUF and safetensors weights listed below.

Based on [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).

**Demo recording:**

https://github.com/user-attachments/assets/218f43db-0406-450d-8a78-5e1d0621f677

## 1. Install

Clone the Demo and enter its root directory:

```bash
git clone https://github.com/APLUX-Pandora/qwen-image-2.1-turbo-demo.git
cd qwen-image-2.1-turbo-demo
```

### Prepare the device

Use an AArch64 Linux device listed in [Model setup](docs/MODEL_SETUP.md#targets), with the required drivers, DSP firmware and FastRPC support. The SDK bundles the GGML runtime and the Hexagon v73 and v75 DSP libraries. LMS activation is not required.

### Install the Demo and SDK

Use Python 3.10–3.12. Create and activate the Demo environment, then install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This installs the SDK, the Web Demo and the Hugging Face CLI. To use only the SDK in your own application, run `python -m pip install ./sdk` in that application's environment.

## 2. Get the model

All supported devices use the same three model files. Download them to `models/` and verify the checksums:

```bash
hf download Abiray/Qwen-Image-2.1-Turbo-GGUF qwen_image_2.1_turbo_Q4_0.gguf --revision 5a4d1964af2bd65111f5f241c65f5b96f0bdd65c --local-dir models
hf download bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF Qwen_Qwen3-VL-8B-Instruct-Q4_0.gguf --revision 6398fcccbd940691854d2cffd85b435ed8eee4ca --local-dir models
hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors --revision df94239739eef7205973a145ffa6f441f7b64e84 --local-dir models
mv models/vae/qwen_image_2.1_vae_bf16.safetensors models/
(cd models && sha256sum -c SHA256SUMS)
```

See [Model setup](docs/MODEL_SETUP.md) for device details.

## 3. Run the Web Demo

```bash
bash demo/run.sh --port 8810
```

Open `http://<device-ip>:8810/`, enter a prompt and select **Generate**. Stop it with `Ctrl+C`. See the [Web Demo guide](demo/README.md).

## 4. Python Usage

In the same environment, from the Demo root:

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

More in the [Python API](sdk/PYTHON_API.md) and [examples](examples/README.md).

## Documentation

- [Model setup](docs/MODEL_SETUP.md): device requirements, targets, download and folder layout
- [Python API](sdk/PYTHON_API.md): every public class, method and parameter
- [C ABI](docs/C_ABI.md): native integration
- [Examples](examples/README.md): runnable Python, C and C++ examples

## License

See [LICENSING.md](LICENSING.md), [NOTICE](NOTICE) and [LICENSES](LICENSES/).
