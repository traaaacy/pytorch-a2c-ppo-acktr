# pytorch-a2c-ppo-acktr

A PyTorch implementation of PPO for use with the pretrained models provided in Assistive Gym.  
This library includes scripts for training and evluating multi agent policies using co-optimization;  
specifically, `main_dual_agent.py` and `enjoy_dual_agent.py`.

## Examples
Refer to [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) for more detailed use cases of this reinforcement learning library.

### Training - Static human
```bash
python3 main.py --env-name "ScratchItchJaco-v0" --algo ppo --use-gae --log-interval 1 --num-steps 200 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 10000000 --use-linear-lr-decay --save-interval 10 --save-dir ./trained_models/
```
We suggest using `nohup` to train policies in the background, disconnected from a terminal instance.
```bash
nohup python3 main.py --env-name "ScratchItchJaco-v0" --algo ppo --use-gae --log-interval 1 --num-steps 200 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 10000000 --use-linear-lr-decay --save-interval 10 --save-dir ./trained_models/ > nohup.out &
```
### Training - Co-optimization, active human and robot
```bash
python3 main_dual_agent.py --env-name "FeedingSawyerHuman-v0" --algo ppo --use-gae --log-interval 1 --num-steps 200 --num-processes 8 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --tau 0.95 --num-env-steps 10000000 --use-linear-lr-decay --save-interval 10 --save-dir ./trained_models/ --action-robot 7 --action-human 4 --obs-robot 25 --obs-human 23
```
### Evaluation - Static human
```
python3 enjoy.py --load-dir trained_models/ppo --env-name "ScratchItchJaco-v0"
```
### Evaluation - Co-optimization, active human and robot
```
python3 enjoy_dual_agent.py --load-dir trained_models/ppo --env-name "FeedingSawyerHuman-v0" --obs-robot 25 --obs-human 23
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

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
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

