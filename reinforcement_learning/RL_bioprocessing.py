import numpy as np
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt


def monod(C, C0, umax, Km, Km0):
    '''
    Calculates the growth rate based on the monod equation

    Parameters:
        C: the concetrations of the auxotrophic nutrients for each bacterial
            population
        C0: concentration of the common carbon source
        Rmax: array of the maximum growth rates for each bacteria
        Km: array of the saturation constants for each auxotrophic nutrient
        Km0: array of the saturation constant for the common carbon source for
            each bacterial species
    '''

    # convert to numpy

    growth_rate = ((umax * C) / (Km + C)) * (C0 / (Km0 + C0))

    return growth_rate

def xdot(x, t, u):
    '''
    Calculates and returns derivatives for the numerical solver odeint

    Parameters:
        S: current state
        t: current time
        Cin: array of the concentrations of the auxotrophic nutrients and the
            common carbon source
        params: list parameters for all the exquations
        num_species: the number of bacterial populations
    Returns:
        xdot: array of the derivatives for all state variables
    '''
    q = 0.5
    y, y0, umax, Km, Km0 = 520000, 480000, 1, 0.00048776, 0.00006845928

    # extract variables

    N, C, C0 = x
    R = monod(C, C0, umax, Km, Km0)

    # calculate derivatives
    dN = N * (R - q)  # q term takes account of the dilution
    dC = q * (u - C) - (1 / y) * R * N
    dC0 = q * (1. - C0) - 1 / y0 * R * N

    # consstruct derivative vector for odeint
    xdot = [dN, dC, dC0]

    return xdot

def reward_func(x, targ):
    '''
    caluclates the reward based on the distance x is from the target
    :param x:
    :param targ:
    :return:
    '''
    N = x[0]
    SE = np.abs(x-targ)
    reward = (1 - sum(SE/targ)/2)/10
    done = False
    if N < 1000:
        reward = - 1
        done = True

    return reward, done

class Environment():

    '''
    Chemostat environment that can handle an arbitrary number of bacterial strains where all are being controlled

    '''

    def __init__(self, xdot, reward_func, u_bounds, pop_bounds, u_disc, pop_disc, sampling_time, scaling):
        '''
        Parameters:
            param_file: path of a yaml file contining system parameters
            reward_func: python function used to coaculate reward: reward = reward_func(state, action, next_state)
            sampling_time: time between sampl-and-hold intervals
            scaling: population scaling to prevent neural network instability in agent, aim to have pops between 0 and 1. env returns populations/scaling to agent
        '''
        self.scaling = scaling

        self.xdot = xdot
        self.xs = []
        self.us = []
        self.sampling_time = sampling_time
        self.reward_func = reward_func

        self.u_bounds = u_bounds
        self.N_bounds = pop_bounds

        self.u_disc = u_disc
        self.N_disc = pop_disc

    def step(self, action):
        '''
        Performs one sampling and hold interval using the action provided by a reinforcment leraning agent

        Parameters:
            action: action chosen by agent
        Returns:
            state: scaled state to be observed by agent
            reward: reward obtained buring this sample-and-hold interval
            done: boolean value indicating whether the environment has reached a terminal state
        '''
        u = self.action_to_u(action)

        #add noise
        #Cin = np.random.normal(Cin, 0.1*Cin) #10% pump noise

        self.us.append(u)

        ts = [0, self.sampling_time]


        sol = odeint(self.xdot, self.xs[-1], ts, args=(u,))[1:]

        self.xs.append(sol[-1,:])

        self.state = self.get_state()

        reward, done = self.reward_func(self.xs[-1])

        return self.state, reward, done

    def get_state(self):
        '''
        Gets the state (scaled bacterial populations) to be observed by the agent

        Returns:
            scaled bacterial populations
        '''
        return self.pop_to_state(self.xs[-1][0])

    def action_to_u(self,action):
        '''
        Takes a discrete action index and returns the corresponding continuous state
        vector

        Paremeters:
            action: the descrete action
            num_species: the number of bacterial populations
            num_Cin_states: the number of action states the agent can choose from
                for each species
            Cin_bounds: list of the upper and lower bounds of the Cin states that
                can be chosen
        Returns:
            state: the continuous Cin concentrations correspoding to the chosen
                action
        '''

        # calculate which bucket each eaction belongs in
        buckets = np.unravel_index(action, [self.u_disc])

        # convert each bucket to a continuous state variable
        u = []
        for r in buckets:
            u.append(self.u_bounds[0] + r*(self.u_bounds[1]-self.u_bounds[0])/(self.u_disc-1))

        u = np.array(u).reshape(1,)

        return np.clip(u, u_bounds[0], u_bounds[1])

    def pop_to_state(self, N):
        '''
        discritises the population of bacteria to a state suitable for the agent
        :param N: population
        :return: discitised population
        '''
        step = (self.N_bounds[1] - self.N_bounds[0])/self.N_disc
        return int(N//step)

    def reset(self,initial_x):
        '''
        Resets env to inital state:

        Parameters:
            initial_S (optional) the initial state to be reset to if different to the default
        Returns:
            The state to be observed by the agent
        '''

        self.xs = [initial_x]
        return self.get_state()

class LT_agent():

    def __init__(self, n_states, n_actions):
        self.gamma = 0.9
        self.n_states = n_states
        self.n_actions = n_actions
        self.values = np.zeros((n_states, n_actions))
        self.alpha = 0.01

    def update_values(self,transition):
        '''
        updates the agents value function based on the experience in transition
        :param transition:
        :return:
        '''
        state, action, reward, next_state, done = transition

        self.values[state, action] += self.alpha * (reward + self.gamma * np.max(self.values[next_state]) * (1-done) - self.values[state, action])

    def get_action(self, state, explore_rate):
        '''
        chooses an action based on the agents value function and the current explore rate
        :param state: the current state given by the environment
        :param explore_rate: the chance of taking a random action
        :return: the action to be taken
        '''
        if np.random.random() < explore_rate:
            action = np.random.choice(range(self.n_actions))


        else:
            action = np.argmax(self.values[state])

        return action

    def get_rate(self, episode, decay, min_r = 0, max_r = 1):
        '''
        Calculates the logarithmically decreasing explore or learning rate

        Parameters:
            episode: the current episode
            MIN_LEARNING_RATE: the minimum possible step size
            MAX_LEARNING_RATE: maximum step size
            denominator: controls the rate of decay of the step size
        Returns:
            step_size: the Q-learning step size
        '''

        # input validation
        if not 0 <= min_r <= 1:
            raise ValueError("MIN_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 <= max_r <= 1:
            raise ValueError("MAX_LEARNING_RATE needs to be bewteen 0 and 1")

        if not 0 < decay:
            raise ValueError("decay needs to be above 0")

        rate = max(min_r, min(max_r, 1.0 - math.log10((episode + 1) / decay)))

        return rate


one_min = 0.016666666667
n_mins = 5

sampling_time = n_mins * one_min # time between actions

u_bounds, pop_bounds, u_disc, pop_disc, scaling = [0, 0.1], [0, 50000], 10, 10, 10000
n_episodes = 1000

t_steps = int((24 * 60) / n_mins)  # set this to 24 hours

n_states = n_actions = 10 # how many state, action pairs to learn values for

target_pop = 35000
reward_f = lambda x : reward_func(x, target_pop)

env = Environment(xdot, reward_f, u_bounds, pop_bounds, u_disc, pop_disc, sampling_time, scaling) # initalise the chemostat environment

initial_x = np.array([20000, 0, 1]) # the initial state

agent = LT_agent(n_states, n_actions)

# train the agent
all_returns = []
for episode in range(n_episodes):
    env.reset(initial_x)
    e_return = 0
    state = env.get_state()
    explore_rate = agent.get_rate(episode, n_episodes/10)

    for i in range(t_steps):
        action = agent.get_action(state, explore_rate)
        next_state, reward, done = env.step(action)
        transition = (state, action, reward, next_state, done)
        agent.update_values(transition)
        state = next_state
        e_return += reward

        if done: # if the bacteria goes extinct
            break

    if episode % 10 == 0:
        print('episode:', episode, ', explore_rate:', explore_rate, ', return:', e_return)


    all_returns.append(e_return)


# plot results
plt.plot(all_returns)
plt.ylabel('Return')
plt.xlabel('Episode')
plt.title('Training performance')

plt.figure()
plt.title('Final population curve')
plt.plot(np.arange(len(env.xs)) *sampling_time, [x[0] for x in env.xs], label = 'population')
plt.hlines(target_pop, 0, 24, label = 'target', color = 'r')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Population cells/L')
plt.show()

