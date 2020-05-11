import numpy as np
from gym import spaces
import optimal_lqr_control
#from stable_baselines.common.env_checker import check_env
import gym
import sys
from scipy.linalg import expm
"""
Settings for linear quadratic regulator
"""
A_rand = np.random.randn(2,2)
A_rand_T = np.transpose(A_rand)
#A2 = expm(A_rand-A_rand_T) 
A2 = np.array([[-0.33629175, -0.94175785],
       [ 0.94175785, -0.33629175]])
#A2 = np.eye(32)
B2 = np.eye(2)
C2 = B2
Q2 = B2
R2 = B2
N2 = 0*B2
#initial_value2 = np.array([[1],[1],[1],[1]])
initial_value2 = np.ones((2,1))
reset_rnd2 = False
nonlin_lambda2 = lambda x: 0.0*np.sin(x)
horizon2 = 20
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
    def __init__(self,A=A2,B=B2,C=C2,Q=Q2,R=R2,N=N2,initial_value=initial_value2, reset_rnd = reset_rnd2, nonlin_lambda = nonlin_lambda2, noise_matrix=0,horizon=horizon2):         
        super(Automatic_Control_Environment, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.noise_matrix = noise_matrix
        self.Q = Q
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
        self.high = 5
        high_vector_act = self.high*np.ones(self.action.shape[0])
        high_vector_obs = self.high*np.ones(self.Y.shape[0])
        self.action_space = spaces.Box(low=-high_vector_act, high=high_vector_act, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_vector_obs, high=high_vector_obs, dtype=np.float32)
        self.nonlin_term = nonlin_lambda

        self.reward_now = 0
        self.rollout_steps = 19
        #self.lqr_optimal = optimal_lqr_control.Lqr(A,B,Q,R,N,horizon)
        
    def state_space_equation(self, action):
        noise = np.random.normal(0,1,self.state.shape)
        #new_state = self.A@self.state+self.B@action+self.noise_matrix*noise
        new_state = self.A@self.state+self.B@action+self.nonlin_term(self.state)+self.noise_matrix*noise

        return new_state

#    def optimal_step(self, state):
#        optimal_action = self.lqr_optimal.action(state)
#        
#        return optimal_action

    def new_obs(self):
        noise = np.random.normal(0,1,self.Y.shape)

        new_Y = self.C@self.state #+ noise
        return new_Y
    
#    def opt_action(self):
#        optimal_action = self.lqr_optimal.action(self.state)
#        optimal_action = np.squeeze(optimal_action,axis=1)
#        return optimal_action

    def step(self, action):
        action = np.expand_dims(action,axis=1)
        #action = np.clip(action,-self.high,self.high)
        next_state = self.state_space_equation(action)
        #next_state = np.clip(next_state,-self.high,self.high)
        self.state = next_state
        self.action = action         
                                                                                                                       
        next_Y = self.new_obs()
        self.Y = next_Y
        reward = self.reward()
        self.reward_now = reward
        

        next_Y = np.squeeze(next_Y,axis=1)
        next_Y = next_Y.astype('float32')
        #next_state = next_state.squeeze()
        #next_state = next_state.astype('float32')
        _ = self.get_debug_dict()
        
        done = self.done()    
        self.nbr_steps += 1 
        #next_Y = np.clip(-self.high,self.high,next_Y)
        return next_Y, self.reward_now, done, _

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
            self.initial_value = np.random.uniform(-0.9,0.9,self.initial_value.shape)

        self.state = self.initial_value
        self.Y = self.new_obs()
        self.action = self.initial_action
        self.nbr_steps = 0
        #self.lqr_optimal.reset()                                                                                                                               

        squeezed_obs = np.squeeze(self.Y,axis=1)
        return squeezed_obs

    def _get_obs(self):
        return self.state

    def reward(self):
        x = self.state
        u = self.action
        x_T = np.transpose(x)
        u_T = np.transpose(u)
        Q = self.Q
        R = self.R
        N = self.N
        current_reward = x_T@Q@x+u_T@R@u+2*x_T@N@x
        return -current_reward[0][0]

    def done(self):
        if self.nbr_steps == self.rollout_steps:
            return True
        #elif np.max(np.abs(self.state)) >= self.high or np.max(np.abs(self.action)) >= self.high:
        #    self.reward_now = self.reward_now + -200+200*self.nbr_steps/self.rollout_steps
        #    return True
        #elif np.max(self.state) > self.high
        else:
            return False


if __name__ == "__main__":
    # A = np.array([[1,0],[0,1]])
    # B = np.array([[1,0],[0,1]])
    # C = np.array([[1,0],[0,1]])
    # Q = np.array([[1,0],[0,1]])
    # R = np.array([[1,0],[0,1]])                                                                                                                    
    # # N = np.array([[0,0],[0,0]])
    #initial_value = np.array([[0.1],[0.1]])
    A = np.array([[0.2,0.3,0.4],[0.1,-0.3,0.4],[0.2,0.5,0.6]])
    B = np.array([[1,0,0],[0,1,0],[0,0,1]])
    C = np.array([[1,1,0],[1,0,0]])
    Q = np.array([[1,0,0],[0,1,0],[0,0,1]])
    R = np.array([[1,0,0],[0,1,0],[0,0,1]]) 
    N = np.array([[0,0,0],[0,0,0],[0,0,0]])
    initial_value = np.array([[0.8],[0.8],[0.8]])
    # A = np.array([[1]])                                                                                                                            
    # # B = np.array([[1]])
    # C = np.array([[1]])
    # Q = np.array([[1]])
    # R = np.array([[1]])
    # N = np.array([[0]])
    # initial_value = np.array([[0.8]])
    ac_env = Automatic_Control_Environment()
    lqr = optimal_lqr_control.Lqr(A2,B2,Q2,R2,N2,20)
    print("obs space: "+str(ac_env.observation_space.shape))
    print("act space: "+str(ac_env.action_space.shape))
    state = ac_env.reset()
    s = []
    a = []
    s.append(state)
    for i in range(20):
        optimal_action = lqr.action(state)
        state, reward, done, _ = ac_env.step(optimal_action)
        s.append(state)
        a.append(optimal_action)
        if i == 19:
            print("s")
       
    
  
         