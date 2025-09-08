"""Setup for installation.
"""

from setuptools import setup

setup(name='gymproxy',
      version='1.0.0',
      authors=['Seungjae Shin','Sae Hyong Park'],
      author_email= ['sjshin0505@{etri.re.kr, gmail.com}', 'labry@etri.re.kr'],
      description=('A python package for porting an external python-based simulation on OpenAI Gym environment using '
                   'multi-threading.'),
      license='BSD',
      packages=['gymproxy'],
      install_requires=['torch==2.8.0','ray[tune]==2.8.1','ray[rllib]==2.8.1','numpy==1.26.4', 'gymnasium==0.28.1'])