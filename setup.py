from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as readme_file:
    long_description = readme_file.read()

setup(
    name="quant",
    version="0.1.0",
    author="Nakul Randad",
    author_email="nakulrandad@gmail.com",
    description="A quantitative finance package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nakulrandad/Quant-Finance",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
