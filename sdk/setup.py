"""Package prebuilt host/DSP assets; never compile an engine or download models."""
from pathlib import Path
import platform,shutil
from setuptools import setup,Distribution
from setuptools.command.build_py import build_py
ROOT = Path(__file__).resolve().parent
RUNTIME_PACKAGE = "_pandora_qwen_image_2_1_turbo_runtime"
class NativeDistribution(Distribution):
    def has_ext_modules(self): return True
class BuildSDK(build_py):
    def run(self):
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("This SDK requires Linux AArch64, FastRPC/BSP and a compatible Hexagon DSP")
        for name in ("libqwen_image.so", "libggml-htp-v73.so", "libggml-htp-v75.so", "LICENSE", "LICENSING.md", "NOTICE", "THIRD_PARTY_NOTICES.txt", "LICENSES"):
            if not (ROOT / "lib" / name).exists(): raise RuntimeError("Missing prebuilt SDK asset: " + name)
        super().run()
        target = Path(self.build_lib) / RUNTIME_PACKAGE / "lib"
        shutil.copytree(ROOT / "lib", target, dirs_exist_ok=True)
setup(packages=[RUNTIME_PACKAGE], py_modules=["pandora_qwen_image_2_1_turbo"], package_dir={"": "python"},
      distclass=NativeDistribution, cmdclass={"build_py": BuildSDK})
