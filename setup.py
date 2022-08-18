import pkg_resources
import os
from setuptools import setup, find_packages

setup(
    name="clip-diffusion",
    py_modules=["clip_diffusion"],
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(requirement)
        for requirement in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
    ],
)
