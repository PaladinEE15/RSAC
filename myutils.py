import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym

from sacae import SACAE
from sacao import SACAO
from rsac import RSAC

RNNModel_Name_Set = ["Prnn","CPDrnn","Wrnn","HPrnn"]
RobModel_Name_Set = ["Prob","CPDrob","Wrob","HProb"]
RawModel_Name_Set = ["Praw","CPDraw","Wraw","HPraw"]
ObsModel_Name_Set = ["Pobs","CPDobs","Wobs","HPobs"]


def getRnnSACresults(env_id, Model_IDs, Result_ID, params, total_dim, episode_len, mem_len=10):
    name = RNNModel_Name_Set[env_id]
    Model_count = len(Model_IDs)
    Noise_count = len(params)
    if env_id == 0:
        env = gym.make('NoisyPendulum-v0')
    elif env_id == 1:
        env = gym.make('DiscreteCartpole-v0')
    elif env_id == 2:
        env = gym.make("Walker2d-v3")
    elif env_id == 3:
        env = gym.make('Hopper-v3')
    Rewards = np.zeros((Model_count*10,Noise_count))
    cct = 0
    obs_dim = env.observation_space.shape[0]
    T_total_len = obs_dim + total_dim
    for outidx in Model_IDs:
        themodel = RSAC.load("Models/"+name+str(outidx))
        for repeattime in range(10):
            for inidx in range(Noise_count):
                if env_id == 0:
                    env.change_env_setting(10, 1, params[inidx])
                elif env_id == 1:
                    env.change_env_settings(0.5*params[inidx])
                elif env_id == 2:
                    mas = env.sim.model.body_mass
                    mas[1] = params[inidx]*3.53429174
                elif env_id == 3:
                    mas = env.sim.model.body_mass
                    mas[1] = params[inidx] * 3.53429174                    
                obs = env.reset()
                sub_reward = 0
                
                total_sequence = []
                virtual_transistion = np.concatenate((np.zeros(total_dim),obs))
                total_sequence.append(virtual_transistion)
                

                for dd in range(episode_len):
                    old_obs = obs
                    [action, _] = themodel.predict(total_sequence)
                    obs, rewards, dones, _ = env.step(action.flatten())
                    obs = obs.flatten()
                    action = action.flatten()            
                    new_record = np.concatenate((old_obs, action, obs))
                    total_sequence.append(new_record)
                    if len(total_sequence) > mem_len :
                        del(total_sequence[0])
                    
                    sub_reward = sub_reward + rewards
                    if dones:
                        break
                Rewards[cct,inidx] = sub_reward
            cct = cct + 1
            print("stage complete:",repeattime)
    RewardsMean = np.mean(Rewards,axis=0)
    RewardsStd = np.std(Rewards, axis=0)    
    np.save("Results/"+name+str(Result_ID)+"Mean.npy",RewardsMean)
    np.save("Results/"+name+str(Result_ID)+"Std.npy",RewardsStd)
    return


def getSACresults(env_id, functype, Model_IDs, Result_ID, params, episode_len):
    if functype == 0: #rob
        name = RobModel_Name_Set[env_id]
    elif functype == 1: #raw
        name = RawModel_Name_Set[env_id]
    else: #obs
        name = ObsModel_Name_Set[env_id]
    Model_count = len(Model_IDs)
    Noise_count = len(params)
    if env_id == 0:
        env = gym.make('NoisyPendulum-v0')
    elif env_id == 1:
        env = gym.make('DiscreteCartpole-v0')
    elif env_id == 2:
        env = gym.make("Walker2d-v3")
    elif env_id == 3:
        env = gym.make('Hopper-v3')
    Rewards = np.zeros((Model_count*10,Noise_count))
    cct = 0
    for outidx in Model_IDs:
        if functype == 0:
            themodel = SACAE.load("Models/"+name+str(outidx))
        else:
            themodel = SACAO.load("Models/"+name+str(outidx))
        for repeattime in range(10):
            for inidx in range(Noise_count):
                if env_id == 0:
                    env.change_env_setting(10, 1, params[inidx])
                elif env_id == 1:
                    env.change_env_settings(0.5*params[inidx])
                elif env_id == 2:
                    mas = env.sim.model.body_mass
                    mas[1] = params[inidx]*3.53429174
                elif env_id == 3:
                    mas = env.sim.model.body_mass
                    mas[1] = params[inidx] * 3.53429174                      
                obs = env.reset()
                sub_reward = 0
                for dd in range(episode_len):
                    [action, _] = themodel.predict(obs)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, rewards, dones, _ = env.step(action)
                    sub_reward = sub_reward + rewards
                    if dones:
                        break
                Rewards[cct,inidx] = sub_reward
            cct = cct + 1
    RewardsMean = np.mean(Rewards,axis=0)
    RewardsStd = np.std(Rewards, axis=0)    
    np.save("Results/"+name+str(Result_ID)+"Mean.npy",RewardsMean)
    np.save("Results/"+name+str(Result_ID)+"Std.npy",RewardsStd)
    return


ColorPool = ['cornflowerblue', 'lightgreen', 'lightsteelblue', 'moccasin']

def MyPlotBar(xticklabels, xname, model_names, labels, pic_name, pic_title,bottom=0):
    xlen = len(xticklabels)
    modelnum = len(model_names)
    std_param = dict(elinewidth=1,ecolor='black',capsize=4)
    bar_width = 0.2
    x = np.arange(xlen)

    for modelidx in range(modelnum):
        name = model_names[modelidx]
        datamean = np.load("Results/"+name+"Mean.npy")/1000
        datamean = datamean - bottom
        datastd = np.load("Results/"+name+"Std.npy")/1000
        plt.bar(x+bar_width*modelidx,datamean,bar_width,color=ColorPool[modelidx],bottom=bottom,yerr=datastd,error_kw=std_param,label=labels[modelidx]) 
           
    plt.title(pic_title,fontsize=18)
    plt.ylabel("Rewards(x1000)",fontsize=14)
    plt.xlabel(xname,fontsize=14)
    plt.xticks(x+bar_width*(modelnum-1)/2,xticklabels)
    plt.legend(fontsize=14,loc="lower left")
    plt.savefig(pic_name, bbox_inches='tight')
    plt.show()
    return()