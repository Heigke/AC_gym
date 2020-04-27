import numpy as np
from gym import spaces
import optimal_lqr_control
#from stable_baselines.common.env_checker import check_env
import gym
import sys
import matplotlib.pyplot as plt

"""
Settings for linear quadratic regulator
"""

A2 = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
B2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[0,0,0]])
C2 = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
Q2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
R2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
N2 = np.array([[0,0,0],[0,0,0],[0,0,0]])
initial_value2 = np.array([[1],[1],[1],[0],[1]])
reset_rnd2 = True
nonlin_lambda2 = lambda x: 0*x
setpoint_freq2 = 50
rollout_steps2 = 49
setpoint_levels2 = [-1,-0.5,0,0.5,1]
setpoint_speeds2 = [1,2,5]
margin2 = 5
punishment2 = 1000
class Automatic_Control_Environment(gym.Env):
    """ ***A simple automatic control environment***
    by Niklas Kotarsky and Eric Bergvall
    
    The system is described by x_t+1 = A*x_t + B*u_t + noise
    The observed system y_t+1 = C*x_t+1 + noise
    where x_t is a column vector with dimension N and A has dimension N x N
    u_t has dimension M and B then have dimension NxM 
    Noise has dimension N and noise_matrix has dimension NxN 
    C has dimensions KxN s.t. y has dimension K. Noise dimension K"""
    

    metadata = {'render.modes': ['human']}
    def __init__(self,
                A=A2,
                B=B2,
                C=C2,
                Q=Q2,
                R=R2,
                N=N2,
                initial_value=initial_value2, 
                reset_rnd = reset_rnd2, 
                nonlin_lambda = nonlin_lambda2, 
                setpoint_freq = setpoint_freq2, 
                rollout_steps = rollout_steps2,
                setpoint_levels = setpoint_levels2,
                setpoint_speeds = setpoint_speeds2,
                margin = margin2,
                punishment = punishment2, 
                noise_matrix=0,
                horizon=100):
        super(Automatic_Control_Environment, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.noise_matrix = noise_matrix
        self.Q = Q
        self.Q_initial = Q
        self.R = R
        self.N = N
        self.reset_rnd = reset_rnd
        self.horizon = horizon
        self.initial_value = initial_value
        self.state = self.initial_value
        self.Y_initial = np.random.uniform(0,1,(self.C.shape[0],1))
        self.Y = self.Y_initial
        self.initial_action = np.random.normal(0,1,(self.B.shape[1],1))
        self.action = self.initial_action
        self.state_limit = 1000
        self.nbr_steps = 0
        self.high = margin
        self.punishment = punishment
        high_vector_act = self.high*np.ones(self.action.shape[0])
        high_vector_obs = self.high*np.ones(self.Y.shape[0])
        self.action_space = spaces.Box(low=-high_vector_act, high=high_vector_act, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_vector_obs, high=high_vector_obs, dtype=np.float32)
        self.nonlin_term = nonlin_lambda
        self.setpoint_freq = setpoint_freq
        self.setpoint_levels = setpoint_levels2
        self.setpoint_speeds = setpoint_speeds2
        self.setpoint = self.initial_value[-2]
        self.setpoint_speed = self.initial_value[-1]
        self.rollout_steps = rollout_steps
        self.lqr_optimal = optimal_lqr_control.Lqr(A,B,Q,R,N,horizon)
        

    def state_space_equation(self, action):
        noise = np.random.normal(0,1,self.state.shape)
        #new_state = self.A@self.state+self.B@action+self.noise_matrix*noise
        new_state = self.A@self.state+self.B@action+self.nonlin_term(self.state)+self.noise_matrix*noise
        
        return new_state

    def new_obs(self):
        noise = np.random.normal(0,1,self.Y.shape)
        
        new_Y = self.C@self.state #+ noise
        return new_Y
    def opt_action(self):
        optimal_action = self.lqr_optimal.action(self.state)
        optimal_action = np.squeeze(optimal_action,axis=1)
        return optimal_action

    def new_setpoint(self):
        if self.nbr_steps+1 % self.setpoint_freq == 0:
            rnd_update = np.random.uniform(0,10)
            rnd_level = np.random.choice(self.setpoint_levels)
            rnd_speed = np.random.choice(self.setpoint_speeds)
            if rnd_update > 5:
                self.setpoint = rnd_level
                self.setpoint_speed = rnd_speed
                return self.setpoint, self.setpoint_speed
        
        return self.setpoint, self.setpoint_speed

    def update_setpoints(self):
        self.state[-1] = self.setpoint_speed
        self.state[-2] = self.setpoint
        return self.state

    def step(self, action):
        new_setpoint, new_setpoint_speed = self.new_setpoint()
        action = np.expand_dims(action,axis=1)
        next_state = self.state_space_equation(action)
        self.state = next_state
        next_state = self.update_setpoints()
        self.state = next_state
        self.action = action
        next_Y = self.new_obs()
        self.Y = next_Y
        reward = self.reward()
        self.nbr_steps += 1
        done, punish = self.done()
        reward = reward - punish
        next_Y = np.squeeze(next_Y,axis=1)
        next_Y = next_Y.astype('float32')
        #next_state = next_state.squeeze()
        #next_state = next_state.astype('float32')
        _ = self.get_debug_dict()
        #next_Y = np.clip(-self.high,self.high,next_Y)
        return next_Y, reward, done, _

    def get_debug_dict(self):
        return dict()


    def render(self, mode='human'):
        nonsense=1
        return
    def close(self):
        nonsense=1
        return

    def observable(self):
        O = []
        for i in range(self.state.shape[0]):
            new_entry = self.C@np.linalg.matrix_power(self.A,i)
            O.append(new_entry)
        O = np.vstack(O)
        rank = np.linalg.matrix_rank(O)
        observable_check = (rank == self.A.shape[0])
        return observable_check

    def reset(self):
        if self.reset_rnd:
            self.initial_value_new = np.random.uniform(-0.9,0.9,self.initial_value.shape)
        
        self.state = self.initial_value_new
        
        
        self.action = self.initial_action
        self.nbr_steps = 0
        self.Q = self.Q_initial
        self.setpoint = self.initial_value[-2]
        self.setpoint_speed = self.initial_value[-1]
        self.state[-2] = self.setpoint
        self.state[-1] = self.setpoint_speed
        self.Y = self.new_obs()
        #self.lqr_optimal.reset()
        squeezed_obs = np.squeeze(self.Y,axis=1)
        return squeezed_obs

    def _get_obs(self):
        return self.state

    def reward(self):
        
        # Remove setpoint and setpoint speeds for reward calculation
        x = self.state[0:-2] 
        u = self.action
        s = np.ones(x.shape)*self.setpoint
        s_T = np.transpose(s)
        x_T = np.transpose(x)
        u_T = np.transpose(u)
        Q = np.eye(self.Q.shape[0])*self.setpoint_speed
        R = self.R
        N = self.N
        current_reward = (x_T-s_T)@Q@(x-s)+u_T@R@u+2*x_T@N@x
        return -current_reward[0][0]

    def done(self):
        x = self.state[0:-2] 
        s = np.ones(x.shape)*self.setpoint
        if self.nbr_steps == self.rollout_steps:
            return True, 0
        #elif np.max(np.abs(x-s)) > self.high:
            return True, self.punishment
        #elif np.max(np.abs(self.action)) > self.high:
            return True, self.punishment
        #elif np.max(self.state) > self.high
        else:
            return False, 0


if __name__ == "__main__":
    # A = np.array([[1,0],[0,1]])
    # B = np.array([[1,0],[0,1]])
    # C = np.array([[1,0],[0,1]])
    # Q = np.array([[1,0],[0,1]])
    # R = np.array([[1,0],[0,1]])
    # N = np.array([[0,0],[0,0]])
    #initial_value = np.array([[0.1],[0.1]])
    A = np.array([[0.2,0.3,0.4],[0.1,-0.3,0.4],[0.2,0.5,0.6]])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])
    C = np.array([[1,1,0],[1,0,0]])
    Q = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    N = np.array([[0,0,0],[0,0,0],[0,0,0]])
    initial_value = np.array([[0.8],[0.8],[0.8]])
    # A = np.array([[1]])
    # B = np.array([[1]])
    # C = np.array([[1]])
    # Q = np.array([[1]])
    # R = np.array([[1]])
    # N = np.array([[0]])
    # initial_value = np.array([[0.8]])
    ac_env = Automatic_Control_Environment()
    print("obs space: "+str(ac_env.observation_space.shape))
    print("act space: "+str(ac_env.action_space.shape))
    while True:
        state = ac_env.reset()
        #optimal_action = ac_env.opt_action()
        state_list = []
        action_list = []
        reward_list = []
        obs_list = []
        for i in range(50):
            if i == 48:
                print("stop")
            action = np.array([0.3,0.3,0.3])
            next_state, reward, done, _ = ac_env.step(action)
            state_list.append(np.squeeze(ac_env.state,axis=1))
            action_list.append(action)
            reward_list.append(reward)
            obs_list.append(next_state)
        
        
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title("states")
        plt.plot(state_list)
        plt.show()
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title("obs")
        plt.plot(obs_list)
        plt.show()
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title("rewards")
        plt.plot(reward_list)
        plt.show()
        
    print("new state")
    print(next_state)
    print("rew")
    print(reward.shape)
    print(done)
    next_state, reward, done, _ = ac_env.step(action)
    print("new state")
    print(next_state)
    print("rew")
    print(reward)
    print(done)
    print(state)
    print(next_state.dtype)
    print(ac_env.observation_space.dtype)
    #check_env(ac_env, warn=True)
    print(ac_env.observable())
    ac_env.opt_action()
  