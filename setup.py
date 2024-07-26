"""Setup for installation.
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

cyslow = Extension(
    "acelerate",
    sources=['./gymproxy/env_proxy_cy.pyx', './gymproxy/base_actual_env_cy.pyx'],
    extra_compile_args=['-O3'],
    extra_link_args=['-O3']
)

setup(name='gymproxy',
      version='1.0.0',
      author='Sae Hyong Park and Seungjae Shin',
      author_email='{labry,sjshin0505}@etri.re.kr',
      description=('A python package for porting an external python-based simulation on OpenAI Gym environment using '
                   'multi-threading.'),
      license='BSD',
      packages=['gymproxy'],
      install_requires=['numpy', 'gymnasium==0.29.1'],
      ext_modules=cythonize([cyslow],
      compiler_directives={
        'cdivision': False,
        'boundscheck': False,
        'wraparound': False,
        'nonecheck': False,
        'initializedcheck': False,
        'embedsignature': True
    })
      )