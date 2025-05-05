from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as readme_file:
    long_description = readme_file.read()

setup(
    name="quant",  # Package name
    version="0.1.0",  # Version number
    author="Nakul Randad",
    author_email="nakulrandad@gmail.com",
    description="A quantitative finance package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nakulrandad/Quant-Finance",
    packages=find_packages(),  # Finds all packages inside the directory
    install_requires=requirements,  # Dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
