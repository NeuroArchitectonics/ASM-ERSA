from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asm_ersa",
    version="4.0.0",
    author="NeuroArchitectonics",
    author_email="1393475440@qq.com",
    description="ASM-ERSA: Addressable Slot Memory + Exact-Recall Sparse Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeuroArchitectonics/ASM-ERSA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "safetensors>=0.3.0",
        "transformers>=4.30.0",
    ],
    entry_points={},
)
