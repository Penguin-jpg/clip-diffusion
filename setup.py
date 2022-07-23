import pkg_resources
import os
from setuptools import setup

setup(
    name="clip-diffusion",
    py_modules=["clip_diffusion"],
    install_requires=[
        str(requirement)
        for requirement in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements", "generation.txt"))
        )
    ],
)
