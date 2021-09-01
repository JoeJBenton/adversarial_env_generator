import gym
from gym.spaces import Tuple, Discrete


class SingleBit(gym.Env):

    env_space = Discrete(2)
    observation_space = Discrete(1)
    action_space = Discrete(2)

    def __init__(self, initial_state):
        # EnvironmentClass.__init__(initial_state)
        self.hidden_bit = initial_state

        self.observation, self.reward, self.done, self.info = 0, 0, False, {}

        self.reset()

    def reset(self):
        self.observation = 0
        self.reward = 0
        self.done = False
        self.info = {}

        return self.observation

    def step(self, action):
        if action == self.hidden_bit:
            self.reward = 1
        else:
            self.reward = 0

        self.done = True

        return [self.observation, self.reward, self.done, self.info]

    def __str__(self):
        return str(self.hidden_bit)

    def partition_function(self, env):
        return str(env)

    def partition_keys(self):
        envs = [0, 1]
        return [self.partition_function(self, env) for env in envs]

    def optimal_partition(self, obs):
        if obs[0] == 1:
            return 0
        else:
            return 1
