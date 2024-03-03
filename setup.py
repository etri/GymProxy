"""Setup for installation.
"""

from setuptools import setup

setup(name='gymproxy',
      version='1.0.0',
      author='Seungjae Shin',
      author_email='sjshin0505@{etri.re.kr, gmail.com}',
      description=('A python package for porting an external python-based simulation on OpenAI Gym environment using '
                   'multi-threading.'),
      license='BSD',
      packages=['gymproxy'],
      install_requires=['numpy', 'gymnasium==0.29.1'])
