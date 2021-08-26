#!/usr/bin/env python

from distutils.core import setup

setup(name='pytb',
      version='1.5',
      author='Sinisa Coh and David Vanderbilt',
      author_email='sinisacoh@gmail.com  dhv@physics.rutgers.edu',
      url='http://www.physics.rutgers.edu/~dhv/pytb',
      py_modules=['pytb'],
      license="gpl-3.0",
      description="Simple solver for tight binding models.",
      long_description="""The tight binding method is an approximate
      approach for solving for the electronic wave functions for
      electrons in solids assuming a basis of localized atomic-like
      orbitals.""",
      platforms=["UNIX","Windows","Mac OS X"]
      )
