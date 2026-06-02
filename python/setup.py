"""tinycml - A pure C11 machine learning library with Python bindings"""

from setuptools import setup, find_packages
import os

setup(
    name="tinycml",
    version="0.1.0",
    description="Pure C11 machine learning library — sklearn-compatible Python bindings",
    long_description=open("python/README.md").read() if os.path.exists("python/README.md") else "",
    long_description_content_type="text/markdown",
    author="Samet Yilmaz Temel",
    url="https://github.com/sametyilmaztemel/tinycml",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "tinycml": ["*.so", "*.dylib", "*.dll"],
    },
    python_requires=">=3.7",
    install_requires=["numpy"],
    extras_require={
        "dev": ["pytest", "scikit-learn"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
