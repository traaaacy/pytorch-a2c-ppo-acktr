from setuptools import setup, find_packages

setup(name='a2c_ppo_acktr',
      packages=find_packages('pytorch_ppo'),
      package_dir={'': 'pytorch_ppo'},
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'pybullet', 'torch', 'torchvision'])

setup(name='ppo',
      packages=find_packages(),
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'pybullet', 'torch', 'torchvision'])

