# Model setup

## Device requirements

- An AArch64 Linux device with the required drivers, DSP firmware and FastRPC support. LMS activation is not required.
- Python 3.10–3.12 in the Demo environment (see [Install](../README.md#1-install)).
- The SDK bundles the GGML runtime and the Hexagon v73 and v75 DSP libraries. See [Targets](#targets) for the library used by each SoC.
- About 9 GiB of storage for the shared model files.

## Targets

Find your device in the table. The SDK uses the device ID to select its target from `models.json`. All four targets share the same files in `models/`.

| SoC | Target folder | Device ID | Bundled DSP library |
| --- | --- | --- | --- |
| QCS8550 | `models/` | 603 | Hexagon v73 |
| QCS9075 | `models/` | 676 | Hexagon v75 |
| QCS8625 | `models/` | 702 | Hexagon v75 |
| QCS8275 | `models/` | 675 | Hexagon v75 |

The SDK detects the DSP version and loads the matching bundled library through FastRPC. If you set `ADSP_LIBRARY_PATH`, include the SDK library directory and preserve the device's firmware search directories.

## Download

From the Demo root, with the Python environment active, use `hf download` to get the three files for your device's target. The commands are the same for all targets because they share the same pinned upstream files:

```bash
hf download Abiray/Qwen-Image-2.1-Turbo-GGUF qwen_image_2.1_turbo_Q4_0.gguf --revision 5a4d1964af2bd65111f5f241c65f5b96f0bdd65c --local-dir models
hf download bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF Qwen_Qwen3-VL-8B-Instruct-Q4_0.gguf --revision 6398fcccbd940691854d2cffd85b435ed8eee4ca --local-dir models
hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors --revision df94239739eef7205973a145ffa6f441f7b64e84 --local-dir models
mv models/vae/qwen_image_2.1_vae_bf16.safetensors models/
(cd models && sha256sum -c SHA256SUMS)
```

## Folder layout

After the download, `models/` looks like this:

```text
models/
  models.json
  SHA256SUMS
  LICENSE-APACHE-2.0.txt
  LICENSE-QWEN-RESEARCH.txt
  qwen_image_2.1_turbo_Q4_0.gguf
  Qwen_Qwen3-VL-8B-Instruct-Q4_0.gguf
  qwen_image_2.1_vae_bf16.safetensors
```

At load time the SDK reads `models.json`, selects the device's target and checks that its model files exist and have the listed sizes. To use a custom location, pass the directory containing `models.json`. For an existing model directory, copy `models.json` and `SHA256SUMS` from the current Demo.

The [upstream resource list](../models/models.json) records the sources and compatible devices. Run the checksum command above to verify the downloaded files against the Demo's pinned checksums.

## Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| Missing model or size mismatch | Complete the three downloads in the folder containing `models.json` and verify the checksums. |
| No model target for this device | Check the device ID against the Targets table and use the supplied `models.json`. |
| SHA256 mismatch | Remove the mismatched file, download it again at the listed revision, and rerun the checksum check. If it still fails, report the mismatch. |
| Hexagon library fails to load | Check that the matching DSP library is present and that DSP firmware and FastRPC work on the device. |
