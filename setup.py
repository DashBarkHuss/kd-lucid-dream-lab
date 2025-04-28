from setuptools import setup, find_packages

setup(
    name="gssc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "brainflow>=5.13.3",
        "numpy>=1.24.3",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "torch>=2.7.0",
        "pytz>=2024.2",
    ],
    python_requires=">=3.9",
) 