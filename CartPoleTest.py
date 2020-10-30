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
params = [0.4,1,1.8,2.6]

batchsize = 64
#train RSAC
for idx in range(1,6):
    env = gym.make('DiscreteCartpole-v0')
    model = RSAC(RnnMlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.set_attack_para(gap=40, thd=0.2, env_id=1, mpc=1.5)
    model.set_choosefrom(20)
    model.set_weight(0.05)
    model.set_memory_length(10)
    model.learn(total_timesteps=100000, log_interval=100)
    model.save("Models/CPDrnn"+str(idx))
    del model
    del env
print("RSAC complete")

#train SAC-AE

for idx in range(1,6):
    env = gym.make('DiscreteCartpole-v0')
    model = SACAE(AEMlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.enable_attack(1)
    model.set_attack_para(gap=40, thd=0.2, env_id=1, mpc=1.5)
    model.set_choosefrom(20)
    model.set_weight(0.05)
    model.learn(total_timesteps=100000, log_interval=100)
    model.save("Models/CPDrob"+str(idx))
    del model
print("SAC-AE complete")
#train SAC 
env = gym.make('DiscreteCartpole-v0')
for idx in range(1,6):
    model = SACAO(MlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=100000, log_interval=100) 
    model.save("Models/CPDraw"+str(idx)) #train and save raw SAC
    del model

#train SAC-AO
env = gym.make('DiscreteCartpole-v0')
for idx in range(1,6):
    model = SACAO(MlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=50000, log_interval=100) 
    model.enable_attack(True)
    model.set_atkparas(0.1,20) #set attack parameters
    model.learn(total_timesteps=50000, log_interval=100)
    model.save("Models/CPDobs"+str(idx)) #train and save SAC-AO
    del model
print("SAC-AO complete")

#test and compare the performance of all methods

myutils.getSACresults(env_id=1, functype=0, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=500) #SAC-AE
myutils.getSACresults(env_id=1, functype=2, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=500) #SAC-AO
myutils.getSACresults(env_id=1, functype=1, Model_IDs=range(1,6), Result_ID=1, params=params, episode_len=500) #SAC
myutils.getRnnSACresults(env_id=1, Model_IDs=range(1,6), Result_ID=1, params=params, total_dim=5, episode_len=500) #RSAC-AE
myutils.MyPlotBar(xticklabels=params, model_names=["CPDr3nn1","CPDrob1","CPDraw1","CPDobs1"], labels=["RSAC-AE","SAC-AE","SAC","SAC-AO"], pic_name="Pics/CartPoleCompare.eps", xname="Length of Pole", pic_title="CartPole")

#extensive experiment: train rawSAC with different parameters

env = gym.make('DiscreteCartpole-v0')
for idx in range(6,11):
    env.change_env_settings(0.5*1.5)
    model = SACAO(MlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=100000) 
    model.save("Models/CPDraw"+str(idx)) #train and save raw SAC
    del model

for idx in range(11,16):
    env.change_env_settings(0.5*2)
    model = SACAO(MlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.enable_attack(False)
    model.learn(total_timesteps=100000) 
    model.save("Models/CPDraw"+str(idx)) #train and save raw SAC
    del model

myutils.getSACresults(env_id=1, functype=1, Model_IDs=range(6,11), Result_ID=2, params=params, episode_len=500) #rawSAC with 1.25
myutils.getSACresults(env_id=1, functype=1, Model_IDs=range(11,16), Result_ID=3, params=params, episode_len=500) #rawSAC with 1.5
myutils.MyPlotBar(xticklabels=params, model_names=["CPDr3nn1","CPDraw1","CPDraw2","CPDraw3"], labels=["RSAC-AE","SAC","SAC-1.5","SAC-2.0"], pic_name="Pics/CartPoleMultiSAC.eps", xname="Length of Pole", pic_title="CartPole")

print('extensive experiment A complete')

#extensive experiment: test randomatk

for idx in range(6,11):
    env = gym.make('DiscreteCartpole-v0')
    model = RSAC(RnnMlpPolicy, env, learning_starts=1000, gamma=0.99, batch_size=batchsize, learning_rate=0.0003)
    model.set_attack_para(gap=40, thd=0.2, env_id=1, mpc=1.5)
    model.enable_rand_attack(1)
    model.set_memory_length(10)
    model.learn(total_timesteps=100000, log_interval=100)
    model.save("Models/CPDrnn"+str(idx))
    del model
    del env

myutils.getRnnSACresults(env_id=1, Model_IDs=range(6,11), Result_ID=2, params=params, total_dim=5, episode_len=500)
myutils.MyPlotBar(xticklabels=params, model_names=["CPDr3nn1","CPDr3nn2","CPDraw1"], labels=["RSAC-AE","RSAC-rand","SAC"], pic_name="Pics/CartPoleRandATK.eps", xname="Length of Pole", pic_title="Cart-Pole")
