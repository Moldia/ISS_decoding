from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.0'
DESCRIPTION = 'Package used to decode ISS data'
LONG_DESCRIPTION = 'This package can be used to decode ISS data'

# Setting up
setup(
    name="ISS_decoding",
    version=VERSION,
    author="Christoffer Mattsson Langseth",
    author_email="<christoffer.langseth@scilifelab.se>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['starfish ','scikit-image == 0.15.0', #starfish ==0.2.1
                    'xarray==0.17.0','tifffile', 
                    'numpy', 'pandas','setuptools',
                    'matplotlib'],
    keywords=['python', 'spatial transcriptomics', 'spatial resolved transcriptomics', 'in situ sequencing', 'ISS','decoding'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
