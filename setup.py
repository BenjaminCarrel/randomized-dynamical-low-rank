import setuptools

with open("README.md", "r") as descr:
    long_description = descr.read()

setuptools.setup(
    name="randomized-dlra",
    version="1.0",
    author="Benjamin Carrel",
    author_email='benjamin.carrel@unige.ch',
    url='https://github.com/BenjaminCarrel/randomized-dynamical-low-rank',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "pip",
        "numpy",
        "scipy",
        "matplotlib",
        "jupyter",
        "tqdm"
    ],
)