import gym
from gym.spaces import Tuple, Discrete


class SequentialMultiBit(gym.Env):

    env_space = Tuple((Discrete(2), Discrete(2), Discrete(2)))
    observation_space = Discrete(1)
    action_space = Discrete(2)

    def __init__(self, initial_state):
        self.hidden_state = initial_state

        self.observation, self.reward, self.done, self.info = 0, 0, False, {}
        self.count = 0

        self.reset()

    def reset(self):
        self.observation = 0
        self.reward = 0
        self.done = False
        self.info = {}

        self.count = 0

        return self.observation

    def step(self, action):
        if action == self.hidden_state[self.count]:
            self.reward = self.count + 1
        else:
            self.reward = 0

        self.count += 1
        if self.count == 3:
            self.done = True

        return [self.observation, self.reward, self.done, self.info]

    def partition_function(self, env):
        return str(env)

    def partition_keys(self):
        envs = []
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    envs.append((x, y, z))
        return [self.partition_function(env) for env in envs]

    def optimal_partition(self, obs):
        # TODO Check this
        if obs[4] == 1:
            return 0
        else:
            return 1
