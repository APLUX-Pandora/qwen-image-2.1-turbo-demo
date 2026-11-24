# Examples

## Overview

| Example | What it shows |
| --- | --- |
| [python/generate.py](python/generate.py) | Generate one PNG with the installed Python SDK. |
| [python/batch_prompts.py](python/batch_prompts.py) | Reuse one loaded model for several prompts. |
| [c/generate.c](c/generate.c) | Generate one PPM using the C ABI and a caller-owned buffer. |
| [cpp/generate.cpp](cpp/generate.cpp) | Use that C ABI with C++ resource ownership. |

`_lifecycle.py` handles coordinated shutdown for the single-image Python example.

## Before you run

Complete [Install](../README.md#1-install) and [Get the model](../README.md#2-get-the-model). Run every command from the Demo root in the activated environment. Stop the Web Demo first; examples and the Web Demo share the NPU.

## Python examples

### generate.py

Generate one 512 × 512 image from the built-in prompt with a fixed seed and save it as `out.png`.

```bash
python examples/python/generate.py
```

Output:

```text
Saved out.png (512 x 512, seed 42)
```

### batch_prompts.py

Generate an image for each prompt in [prompts.txt](prompts.txt).

```bash
python examples/python/batch_prompts.py
find out/batch -name '*.png' | sort
```

Output:

```text
out/batch/00_42.png
out/batch/01_42.png
out/batch/02_42.png
```

| Option | Default | Purpose |
| --- | --- | --- |
| Prompt file | `examples/prompts.txt` | Text file with one prompt per line. |
| `--tier`, `--aspect` | `512` for the default prompt file, `1024` with a supplied file, and aspect ratio `1:1` | Image size preset. Supported tiers are listed by the SDK. |
| `--seed` | `42` | Set the seed. |
| `--models` | `models/` | Select a different model manifest directory. |
| `--seeds` | `1` | Number of images per prompt with consecutive seeds. |
| `--out-dir` | `out/batch` | Output folder. |

## C example

### generate.c

Generate a PPM image with a fixed prompt, 512 × 512 dimensions and seed 42.

Build against the headers and prebuilt library in `sdk/`:

```bash
cc -std=c11 -Wall -Wextra -Werror -pthread -I sdk/include examples/c/generate.c -L sdk/lib -lqwen_image -Wl,-rpath,'$ORIGIN/../../sdk/lib' -o examples/c/generate
examples/c/generate models c.ppm
file c.ppm
```

Output:

```text
c.ppm: Netpbm image data, size = 512 x 512, rawbits, pixmap
```

### generate.cpp

Generate the same image using the C ABI from C++.

```bash
c++ -std=c++17 -Wall -Wextra -Werror -pthread -I sdk/include examples/cpp/generate.cpp -L sdk/lib -lqwen_image -Wl,-rpath,'$ORIGIN/../../sdk/lib' -o examples/cpp/generate
examples/cpp/generate models cpp.ppm
file cpp.ppm
```

Output:

```text
cpp.ppm: Netpbm image data, size = 512 x 512, rawbits, pixmap
```

See [C ABI](../docs/C_ABI.md) for ownership and error handling.
