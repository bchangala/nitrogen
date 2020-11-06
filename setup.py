import setuptools
from nitrogen import __version__

with open("README.md", "r") as fh:
	long_description = fh.read()
	
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
	python_requires = '>=3.6',
    install_requires=['numpy>=1.19',
                      'scipy>=1.4.1']
)