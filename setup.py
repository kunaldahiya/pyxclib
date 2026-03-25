from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "xclib.utils._sparse",
        ["xclib/utils/_sparse.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(ext_modules=cythonize(extensions))

