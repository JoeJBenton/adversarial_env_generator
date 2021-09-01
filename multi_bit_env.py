import gym
from gym.spaces import Tuple, Discrete


class MultiBit(gym.Env):

    env_space = Tuple((Discrete(2), Discrete(2), Discrete(2)))
    observation_space = Discrete(1)
    action_space = Tuple((Discrete(2), Discrete(2), Discrete(2)))

    def __init__(self, initial_state):
        self.hidden_state = initial_state

        self.observation, self.reward, self.done, self.info = 0, 0, False, {}

        self.reset()

    def reset(self):
        self.observation = 0
        self.reward = 0
        self.done = False
        self.info = {}

        return self.observation

    def step(self, action):
        reward_vector = [1, 2, 3]
        self.reward = 0

        for guess, reward, hidden_bit in zip(action, reward_vector, self.hidden_state):
            if guess == hidden_bit:
                self.reward += reward

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
        return [self.partition_function(self, env) for env in envs]

    def optimal_partition(self, obs):
        # TODO Check this
        if obs[4] == 1:
            return 0
        else:
            return 1
