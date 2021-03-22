import os
import matplotlib.pyplot as plt

from gym_minigrid.wrappers import *
from gym_minigrid.wrappers import FlatObsWrapper
from stable_baselines import PPO2
from stable_baselines.common.vec_env import  DummyVecEnv

tensorboard_folder = './tensorboard/MiniGrid-DoorKey-16x16-v0/'
model_folder = './models/MiniGrid-DoorKey-16x16-v0/'
if not os.path.isdir(tensorboard_folder):
    os.makedirs(tensorboard_folder)
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

env = gym.make('MiniGrid-DoorKey-16x16-v0')
env = FlatObsWrapper(env)
model = PPO2('MlpPolicy', env, verbose=0, nminibatches=1, n_steps=128,tensorboard_log=tensorboard_folder)
model.learn(total_timesteps=10000, tb_log_name='PPO2')