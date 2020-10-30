import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math


class NoisySelfnavEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    #action: turn left or right
    #observe: 7 distances, speed direction, target distance, target direction, x, y 
    #direction is [-1,1],corresbonding to [-pi,pi]
    #the detection range should be [0,20], and we normalize it to [-1,1]
    #the x and y scale should be [0,120] and [0,60], and we normalize it to [-2,2] and [-1,1]
    #the distance to target should be [0,200], we normalize it to [-1,1]

    def __init__(self):
        self.max_detec_dis=1
        self.min_detec_dis=0
        self.max_direction=1
        self.min_direction=-1
        self.max_targ_dis=1
        self.min_targ_dis=0

        self.max_action=0.5 #can only turn 90
        self.min_action=-0.5

        self.low_state=np.array([self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,\
        self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,self.min_detec_dis,\
        self.min_direction,self.min_targ_dis,self.min_direction],dtype=np.float32)

        self.high_state=np.array([self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,\
        self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,self.max_detec_dis,\
        self.max_direction,self.max_targ_dis,self.max_direction],dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        #initialize a set of hyper-parameters
        #spawn locations
        self.easy_high = 60
        self.easy_low = 50
        self.hard_high = 40
        self.hard_low = 30
        self.start_x = 0
        self.target_x = 120
        #obstacle size
        self.obs_height = 40
        #reward parameters
        self.trans_r = 2
        self.obs_p1 = 8
        self.obs_p2 = 5
        self.step_p = -0.6
        self.suc_r = 100
        self.fail_p = 0

        self.seed()
        self.reset()

    def set_start_loc (self, easy_high, easy_low, hard_high, hard_low):
        self.easy_high = easy_high
        self.easy_low = easy_low
        self.hard_high = hard_high
        self.hard_low = hard_low

    def set_x (self, start_x, target_x):
        self.start_x = start_x
        self.target_x = target_x
    
    def set_reward_para (self, trans_r, obs_p1, obs_p2, step_p, suc_r, fail_p):
        self.trans_r = trans_r
        self.obs_p1 = obs_p1
        self.obs_p2 = obs_p2
        self.step_p = step_p
        self.suc_r = suc_r
        self.fail_p = fail_p  
    
    def set_barrier_height(self, obs_height):
        self.obs_height = obs_height

    def getdistance(self,x,y,direc):
        delta_x=0.1*np.cos(direc*math.pi)
        delta_y=0.1*np.sin(direc*math.pi)
        #recover normal axis
        cur_x = x
        cur_y = y
        for ccount in range(200):
            cur_x=cur_x+delta_x
            cur_y=cur_y+delta_y
            if cur_y>=60 and cur_y<=0:
                return ccount*0.005
            if cur_y<=self.obs_height:
                if cur_x>=10 and cur_x<=50:
                    return ccount*0.005
            if cur_y>=60-self.obs_height:
                if cur_x>=70 and cur_x<=110:
                    return ccount*0.005
        return 1

    def reset(self,mode=0):
        #mode=0,easy; mode=1, hard
        if mode == 0:
            self.start_loc = np.array([self.start_x,self.np_random.uniform(low=self.easy_low,high=self.easy_high)])
            self.target_loc = np.array([self.target_x,self.np_random.uniform(low=60-self.easy_high,high=60-self.easy_low)])
        else :
            self.start_loc = np.array([self.start_x,self.np_random.uniform(low=self.hard_low,high=self.hard_high)])
            self.target_loc = np.array([self.target_x,self.np_random.uniform(low=60-self.hard_high,high=60-self.hard_low)])
        self.cur_x_real = self.start_loc[0]
        self.cur_y_real = self.start_loc[1]
        self.speed = 2
        self.speed_direc = 0
        self.x_dist = self.target_loc[0]-self.start_loc[0]
        self.y_dist = self.target_loc[1]-self.start_loc[1]
        self.real_dist = math.sqrt(self.x_dist**2+self.y_dist**2)
        self.old_dist = self.real_dist 
        normed_dist = 2/(np.exp(-0.002*self.real_dist) + 1)-1
        self.target_direc = math.atan(self.y_dist/self.x_dist)/math.pi
        if self.x_dist < 0 :
            self.target_direc = -self.target_direc
        self.state = np.array([1,1,1,1,1,1,1,self.speed_direc,normed_dist,self.target_direc])
        self.detect_direc = [1/2,1/3,1/6,0,-1/6,-1/3,-1/2]
        return self.state

    def seed(self, seed=None): #i don't know what's this for actually
        self.np_random, seed =seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #check action range
        if action > self.max_action :
            action = self.max_action
        if action < self.min_action :
            action = self.min_action
        #perform move first
        self.speed_direc=angle_norm(self.state[7]+action)
        self.cur_x_real=self.cur_x_real+self.speed*math.cos(math.pi*self.speed_direc)
        self.cur_y_real=self.cur_y_real+self.speed*math.sin(math.pi*self.speed_direc)
        #get observations
        self.x_dist=self.target_loc[0]-self.cur_x_real
        self.y_dist=self.target_loc[1]-self.cur_y_real
        self.target_direc=math.atan(self.y_dist/self.x_dist)/math.pi
        if self.x_dist < 0 :
            self.target_direc = -self.target_direc
        self.old_dist = self.real_dist
        self.real_dist = math.sqrt(self.x_dist**2+self.y_dist**2)
        #detect obstacles. this can be parallized later
        kk=0
        for bias in self.detect_direc:
            direc = angle_norm(self.speed_direc+bias)
            self.state[kk] = self.getdistance(self.cur_x_real,self.cur_y_real,direc)
            kk = kk+1
        #renew other states
        self.state[7] = self.speed_direc
        self.state[8] = 2/(np.exp(-0.002*self.real_dist) + 1)-1
        self.state[9] = self.target_direc
        #judge whether done
        if self.real_dist < 10 :
            isarrival = True
        else :
            isarrival = False
        #calculate reward
        #obstacle penalty
        min_dis = 20*np.min(self.state[0:7])
        if min_dis > 0 :
            obs_pny = -self.obs_p1*np.exp(-self.obs_p2*min_dis)
            iscrash = False
        else :
            obs_pny = self.fail_p
            iscrash = True
        #step penalty
        sep_pny = self.step_p
        #transition reward
        trans_reward = self.trans_r*(self.old_dist-self.real_dist)
        #success reward
        if isarrival:
            success_reward = self.suc_r
        else:
            success_reward = 0
        isdone = isarrival or iscrash
        total_reward = obs_pny+sep_pny+trans_reward+success_reward

        return self.state, total_reward , isdone, {}

    
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(1200,600)
            obs_a_loc = [[10,0],[10,40],[50,0],[50,40]]
            obs_b_loc = [[70,20],[70,60],[110,20],[110,60]]
            self.obs_a = rendering.make_polygon(obs_a_loc)
            self.obs_b = rendering.make_polygon(obs_b_loc)
            self.viewer.add_geom(self.obs_a)
            self.viewer.add_geom(self.obs_b)

            car = rendering.make_circle(2)
            car.add_attr(rendering.Transform())
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
        
        pos_x = self.cur_x_real
        pos_y = self.cur_y_real
        self.cartrans.set_translation(pos_x,pos_y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




def angle_norm(x):
    if x>1:
        x=x-2
    if x<-1:
        x=x+2
    return x


            
