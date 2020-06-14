from setuptools import setup, find_packages

setup(name='ppo',
      packages=find_packages(),
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'torch', 'torchvision', 'baselines @ git+git://github.com/openai/baselines.git@master#egg=baselines'])
