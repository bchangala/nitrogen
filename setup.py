import setuptools
from setuptools import Extension
import codecs
import os

####################
# Version fetching
#
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
        
__version__ = get_version("nitrogen/__init__.py")
        
#######################
# Cython handling
# (adapted from C. McQueen `simplerandom`
#  see https://github.com/cmcqueen/simplerandom)
# True -> build extensions using Cython
# False -> build extensions from C file
# 'auto' -> build with Cython if available, otherwise from C
#
# To build cython code in-place from source
# (e.g. with a clone of the source repository)
# run setuptools as
#
# >> python setup.py build_ext --inplace
#
# To force a complete re-build, run 
#
# >> python setup.py build_ext --inplace --force
#
use_cython = True #'auto'
#
#
if use_cython:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if use_cython == 'auto':
            use_cython = False 
        else:
            raise # use_cython was True, but cannot import 
            
cmdclass = {}
ext_modules = []
if use_cython:
    
    ext_modules += [
        Extension("nitrogen.cythontest", [ "nitrogen/cython/test.pyx" ]),
        Extension("nitrogen.basis.ndbasis_c", [ "nitrogen/basis/cython/ndbasis_c.pyx" ]),
        Extension("nitrogen.autodiff.cyad.cyad_core", ["nitrogen/autodiff/cyad/cyad_core.pyx"]),
        Extension("nitrogen.pes.library.mal_mht2014.malpes", ["nitrogen/pes/library/mal_mht2014/malpes.pyx"]),
        Extension("nitrogen.pes.fit.exppip", ["nitrogen/pes/fit/cython/exppip.pyx"]),
    ]
    
    cmdclass.update({ 'build_ext': build_ext })
    
else:
    ext_modules += [
        Extension("nitrogen.cythontest", [ "nitrogen/cython/test.c" ]),
        Extension("nitrogen.basis.ndbasis_c", [ "nitrogen/basis/cython/ndbasis_c.c" ]),
        Extension("nitrogen.autodiff.cyad.cyad_core", ["nitrogen/autodiff/cyad/cyad_core.c"]),
        Extension("nitrogen.pes.library.mal_mht2014.malpes", ["nitrogen/pes/library/mal_mht2014/malpes.c"]),
        Extension("nitrogen.pes.fit.exppip", ["nitrogen/pes/fit/cython/exppip.c"]),
        
    ]
    
#for e in ext_modules:
#    e.cython_directives = {'language_level': "3"} #all are Python-3
    
######################################


with open("README.md", "r") as fh:
	long_description = fh.read()
    
on_rtd = os.environ.get('READTHEDOCS') == 'True' # Check whether we are on Read the Docs
install_requires = ['numpy>=1.19', 'scipy>=1.4.1', 'matplotlib>=3.1,<3.3', 'scikit-image',
                  'setuptools>=49',"wheel","Cython>=0.29.21"]
if not on_rtd:
    # py3nj requires fortran compilation; cannot be built on read-the-docs
    install_requires += ['py3nj']
	
setuptools.setup(
	name = "nitrogen",
	version = __version__,
	author = "Bryan Changala",
	author_email = "bryan.changala@cfa.harvard.edu",
	description = "A scientific computing package for nuclear motion calculations of small molecules.",
	long_description = long_description,
	long_description_content_type = "text/markdown",
	url = "https://github.com/bchangala/nitrogen",
	packages = setuptools.find_packages(),
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires = '>=3.7.11', 
    install_requires=install_requires,
    cmdclass = cmdclass,
    ext_modules = ext_modules,
)