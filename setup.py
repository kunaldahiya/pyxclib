import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    Extension(
        "xclib.data._sparse",
        ["xclib/data/_sparse.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]

setuptools.setup(
    name="xclib",
    version="0.95",
    author="X",
    author_email="kunalsdahiya@gmail.com/anshumitts@gmail.com",
    description="An extreme classification library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/kunaldahiya/xclib",
    packages=setuptools.find_packages(),
    package_data={'xclib': ["classifier/so/*.so"]},
    ext_modules = cythonize(extensions),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

