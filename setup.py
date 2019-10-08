from setuptools import setup, find_packages

setup(name='pytorch_ppo',
      packages=find_packages(),
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'pybullet', 'torch', 'torchvision'])
