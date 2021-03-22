import os
import matplotlib.pyplot as plt

from gym_minigrid.wrappers import *
from gym_minigrid.wrappers import FlatObsWrapper
from stable_baselines import PPO2
from stable_baselines.common.vec_env import  DummyVecEnv

tensorboard_folder = '/root/code/stable_baselines/tensorboard/MiniGrid-Empty-16x16/'
model_folder = './models/MiniGrid-Empty-16x16/'
if not os.path.isdir(tensorboard_folder):
    os.makedirs(tensorboard_folder)
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

env = gym.make('MiniGrid-Empty-16x16-v0')
env = FlatObsWrapper(env)
model = PPO2('MlpPolicy', env, verbose=0, nminibatches=1, n_steps=128,tensorboard_log=tensorboard_folder)
model.learn(total_timesteps=1000000, tb_log_name='PPO2')
model.save(model_folder + "PPO2")
