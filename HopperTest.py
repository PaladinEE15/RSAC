#train RSAC-AE 

from rsac import RSAC
from rsacpolicies import RnnMlpPolicy
from sacae import SACAE
from aepolicies import AEPolicy, AEMlpPolicy
from sacao import SACAO
from stable_baselines.sac.policies import MlpPolicy
import gym
import numpy as np
import myutils

for idx in range(1,6):
    env = gym.make('Hopper-v3')
    model = RSAC(RnnMlpPolicy, env, learning_starts=10000, batch_size=64, buffer_size=500000, gamma=0.99, learning_rate=0.0003)
    model.set_attack_para(gap=40, thd=0.75, env_id=3, mpc=1.5)
    model.set_choosefrom(20)
    model.set_weight(0.2)
    model.set_memory_length(10)
    model.learn(total_timesteps=2000000)
    model.save("Models/HPrnn"+str(idx))
    del model
    del env
    print("RSAC complete")

    #train SAC-AE
    env = gym.make('Hopper-v3')
    model = SACAE(AEMlpPolicy, env, learning_starts=10000, batch_size=64, buffer_size=500000, ent_coef=0.01, learning_rate=0.0003)
    model.enable_attack(1)
    model.set_attack_para(gap=40, thd=0.75, env_id=3, mpc=1.5)
    model.set_choosefrom(20)
    model.set_weight(0.2)
    model.learn(total_timesteps=2000000)
    model.save("Models/HProb"+str(idx))
    del model
    print("SAC-AE complete")

    #train SAC 
    env = gym.make('Hopper-v3')
    model = SACAO(MlpPolicy, env, learning_starts=10000, batch_size=64, buffer_size=500000, gamma=0.99, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=2000000) 
    model.save("Models/HPraw"+str(idx)) #train and save raw SAC

    #train SAC-AO
    env = gym.make('Hopper-v3')
    model = SACAO(MlpPolicy, env, learning_starts=10000, batch_size=64, buffer_size=500000, gamma=0.99, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=1000000) 
    model.enable_attack(True)
    model.set_atkparas(0.008,20) #set attack parameters
    model.learn(total_timesteps=1000000)
    model.save("Models/HPobs"+str(idx)) #train and save SAC-AO
    del model
    print("SAC-AO complete")


import myutils
params = [0.6,1,1.4,1.8]
myutils.getSACresults(env_id=3, functype=0, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=1000) #SAC-AE
myutils.getSACresults(env_id=3, functype=2, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=1000) #SAC-AO
myutils.getSACresults(env_id=3, functype=1, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=1000) #SAC
myutils.getRnnSACresults(env_id=3, Model_IDs=range(1,6), Result_ID=1, params=params, total_dim=14, episode_len=1000, mem_len=10) #RSAC-AE
myutils.MyPlot(xticklabels=params, model_names=["HPrnn1","HProb1","HPraw1","HPobs1"], labels=["RSAC","SACAE","SAC","SACAO"], pic_name="Pics/HopperCompareFinal.eps", xname="Mass of Torso", pic_title="Hopper")




