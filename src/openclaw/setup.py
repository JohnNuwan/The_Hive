from setuptools import find_packages, setup

SUBPACKAGES = find_packages(where=".")

setup(
    name="openclaw",
    version="0.1.0",
    packages=["openclaw", *[f"openclaw.{pkg}" for pkg in SUBPACKAGES]],
    package_dir={"openclaw": "."},
    install_requires=[],
    description="Internal package for RLM capabilities",
)
