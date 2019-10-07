# pytorch-a2c-ppo-acktr

A PyTorch implementation of PPO for use with the pretrained models provided in [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym).  
This library includes scripts for training and evluating multi agent policies using co-optimization; specifically, `main_dual_agent.py` and `enjoy_dual_agent.py`.

## Installation and pretrained models
This library and the pretrained policies for Assistive Gym can be downloaded using the following:
```bash
pip3 install pytorch torchvision tensorflow
# Install OpenAI Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip3 install .
cd ../
# Install pytorch RL library
git clone https://github.com/Zackory/pytorch-a2c-ppo-acktr
cd pytorch-a2c-ppo-acktr
pip3 install .
# Download pretrained policies
cd trained_models/ppo
wget -O pretrained_policies.zip https://goo.gl/Xjh6x4
unzip pretrained_policies.zip
rm pretrained_policies.zip
cd ../../
```

## Examples
Refer to [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) for more detailed use cases of this reinforcement learning library.

### Training - Static human
```bash
python3 main.py --env-name "ScratchItchJaco-v0" --num-env-steps 10000000
```
We suggest using `nohup` to train policies in the background, disconnected from the terminal instance.
```bash
nohup python3 main.py --env-name "ScratchItchJaco-v0" --num-env-steps 10000000 --save-dir ./trained_models/ > nohup.out &
```
See [arguments.py](https://github.com/Zackory/pytorch-a2c-ppo-acktr/blob/master/a2c_ppo_acktr/arguments.py) for a full list of available arguments and hyperparameters.

### Training - Co-optimization, active human and robot
```bash
python3 main_dual_agent.py --env-name "FeedingSawyerHuman-v0" --num-env-steps 10000000
```
### Evaluation - Static human
```
python3 enjoy.py --env-name "ScratchItchJaco-v0"
```
### Evaluation - Co-optimization, active human and robot
```
python3 enjoy_dual_agent.py --env-name "FeedingSawyerHuman-v0"
```

## pytorch-a2c-ppo-acktr-gail
This library is derived from code by Ilya Kostrikov: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```
