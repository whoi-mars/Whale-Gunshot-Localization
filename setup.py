from setuptools import setup, find_packages

with open("requirements.txt") as f:
      requirements = f.read().splitlines()

setup(name='whale_gunshot_localization',
      version='1.0.0',
      author='Mark Goldwater',
      author_email='mgoldwater@whoi.edu',
      packages=find_packages(),
      install_requires=requirements)