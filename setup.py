from pathlib import Path

from setuptools import find_packages, setup

_version=0.1

setup(
    name="visualcla",
    version=_version,
    packages=find_packages('models'),
    package_dir={'':'models'},
    install_requires=[
        "transformers >= 4.29.0",
        "torch",
        "peft",
        "Pillow",
        "sentencepiece",
        "numpy"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)