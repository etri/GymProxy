"""Setup for installation.
"""

from setuptools import setup

setup(name='gymproxy',
      version='1.0.0',
      author='Sae Hyong Park and Seungjae Shin',
      author_email='{labry,sjshin0505}@etri.re.kr',
      description=('A python package for porting an external python-based simulation into OpenAI Gym environment using '
                   'multi-threading.'),
      license='BSD',
      packages=['gymproxy'],
      install_requires=['numpy', 'gymnasium==0.29.1'])
