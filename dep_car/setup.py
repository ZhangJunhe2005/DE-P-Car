#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="dep-car-core",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=["numpy>=1.17"],
)

