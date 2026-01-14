from setuptools import setup, find_packages

setup(
    name="mllib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    extras_require={
        "gpu": ["cupy-cuda12x"]
    }
)
