import sys

from setuptools import setup

setup(name='kernuller',
      version='0.1.0', # defined in the __init__ module
      description='A python toolbox for designing an modelling kernel-nulling interferometric combiner for high contrast interferometric observation of astrophysical sources.',
      url='http://github.com/rlaugier/kernuller',
      author='Romain Laugier',
      author_email='romain.laugier@oca.eu',
      license='',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Professional Astronomers',
          'Topic :: High Angular Resolution Astronomy :: Interferometry',
          'Programming Language :: Python :: 3.7'
      ],
      packages=['kernuller'],
      install_requires=[
          'numpy', 'sympy', 'scipy', 'matplotlib', 'astropy','tqdm', 'astropy'
      ],
      include_package_data=True,
      zip_safe=False)

