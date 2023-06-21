import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


with open("README.md", "r") as fh:
    long_description = fh.read()


extensions = [
    Extension(
        "xclib.utils._sparse",
        ["xclib/utils/_sparse.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]
# cpp_module = [ Extension('xclib.classifier.so.parabel',
#                     include_dirs = ['-pthread', os.environ['EIGEN'], os.environ['PYBIND']],
#                     extra_compile_args = ["-std=c++11"],
#                     library_dirs = ['/usr/local/lib'],
#                     sources=['xclib/classifier/pyParabel/parabel.cpp']),]

setuptools.setup(
    name="xclib",
    version="0.97",
    author="X",
    author_email="kunalsdahiya@gmail.com/me@anshulmittal.org",
    description="An extreme classification library for python",
    long_description_content_type="text/markdown",
    url="https://github.com/kunaldahiya/xclib",
    install_requires=['numpy', 'nmslib', 'scikit-learn', 'numba', 'fasttext'],
    packages=setuptools.find_packages(),
    # package_data={'xclib': ["classifier/so/*.so"]},
    ext_modules=cythonize(extensions),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

