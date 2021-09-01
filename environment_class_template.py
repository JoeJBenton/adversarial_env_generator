import gym
from gym.spaces import Tuple, Discrete


class EnvironmentTemplate(gym.Env):

    # A gym space encoding the set of possible environments
    env_space = None

    # The observation space of the antagonist/protagonist
    observation_space = None

    # The action space of the antagonist/protagonist
    action_space = None

    def __init__(self, initial_state):
        # Creates an instance of the environment class with a given initial_state
        # initial_state must be an element of env_space

        self.observation, self.reward, self.done, self.info = None, None, False, {}

        self.reset()

    def reset(self):
        # Resets the environment into the same initial state

        self.observation = 0
        self.reward = 0
        self.done = False
        self.info = {}

        return self.observation

    def step(self, action):
        # Performs a step of the environment

        return [self.observation, self.reward, self.done, self.info]

    def __str__(self):
        # Creates a string rendering of the environment

        return ""

    def partition_function(self, env):
        # A map from elements of env_space to strings
        # This serves to define a partition the environment space into several classes which is used for logging

        return str(env)

    def partition_keys(self):
        # Returns the range of partition_function; this is used in the creation of logs

        return [self.partition_function(self, env) for env in self.env_space]

    def optimal_partition(self, obs):
        # A map from elements of env_space to {0, 1} used to define the PerfectDescriber policy
        # Note that obs will be formatted using a OneHotEncoder

        return 0
