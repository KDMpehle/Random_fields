import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name = "GaussRF"
    version = "0.0.1",
    author = "K. Mpehle, S. Maharaj"
    author_email = "khaya.mpehle@gmail.com",
    description = "simulation of 1-D and 2-D random fields"
    long_description = long_description,
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3 ",
        "Licence :: OSI :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires = '>=3.2',
    )
