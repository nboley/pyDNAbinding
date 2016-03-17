"""
Copyright (c) 2011-2015 Nathan Boley

This file is part of pyDNAbinding.

pyDNAbinding is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

pyTFbindtools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyTFbindtools.  If not, see <http://www.gnu.org/licenses/>.
"""
from distutils.core import setup, Extension

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = cythonize([
    Extension("pyDNAbinding.sequence", 
              ["pyDNAbinding/sequence.pyx", ]),
])

config = {
    'include_package_data': True,
    'ext_modules': extensions, 
    'description': 'pyDNAbinding',
    'author': 'Nathan Boley',
    'url': 'NA',
    'download_url': 'https://github.com/nboley/pyDNAbinding/',
    'author_email': 'npboley@gmail.com',
    'version': '0.1.1',
    'packages': ['pyDNAbinding', ],
    'setup_requires': [],
    'install_requires': [ 'scipy', 'numpy', 'psycopg2' ],
    'scripts': [],
    'name': 'pyDNAbinding'
}

if __name__== '__main__':
    setup(**config)
