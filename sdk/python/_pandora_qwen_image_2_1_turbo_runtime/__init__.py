"""Installed native assets; model weights and FastRPC/BSP remain external."""
from pathlib import Path
def library_path():
    root = Path(__file__).resolve().parent / "lib"
    for name in ("libqwen_image.so", "libggml-htp-v73.so", "libggml-htp-v75.so"):
        if not (root / name).is_file():
            raise FileNotFoundError("Missing installed SDK asset: " + name)
    return root / "libqwen_image.so"
