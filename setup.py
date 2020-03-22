from setuptools import setup, find_packages

setup(name='a2c_ppo_acktr',
      packages=find_packages('ppo'),
      package_dir={'': 'ppo'},
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'torch', 'torchvision', 'baselines-0.1.6 @ git+https://github.com/openai/baselines@master'])

setup(name='ppo',
      packages=find_packages(),
      version='0.1.0',
      install_requires=['gym', 'matplotlib', 'torch', 'torchvision', 'baselines-0.1.6 @ git+https://github.com/openai/baselines@master'])
