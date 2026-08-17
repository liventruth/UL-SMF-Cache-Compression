from setuptools import setup, find_packages

setup(
    name="ul_smf",
    version="1.0.0",
    description="Unified Latent-State Memory Fabric",
    py_modules=["ul_smf"],
    install_requires=[
        "torch>=2.0.0",
    ],
)
