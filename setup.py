import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xclib",
    version="0.95",
    author="X",
    author_email="kunalsdahiya@gmail.com/anshumitts@gmail.com",
    description="An extreme classification library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/kunaldahiya/xclib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

