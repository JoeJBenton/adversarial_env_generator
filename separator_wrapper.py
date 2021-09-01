from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Tuple, Discrete


class SeparatorEnv(MultiAgentEnv):

    randomness_space = Tuple((Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2)))

    def __init__(self, environment_class, double_antagonist=False, config=None):
        if double_antagonist:
            self.double_antagonist = True
            self.agents = ["antagonist_0", "antagonist_1", "protagonist", "builder", "describer"]
        else:
            self.double_antagonist = False
            self.agents = ["antagonist", "protagonist", "builder", "describer"]

        self.environment_class = environment_class

        self.builder_obs_space = self.randomness_space
        self.describer_obs_space = Tuple((self.environment_class.env_space, self.builder_obs_space))
        if double_antagonist:
            self.antagonist_obs_space = self.environment_class.observation_space
        else:
            self.antagonist_obs_space = Tuple((self.environment_class.observation_space, Discrete(2)))
        self.protagonist_obs_space = self.environment_class.observation_space

        self.builder_action_space = self.environment_class.env_space
        self.describer_action_space = Discrete(2)
        self.antagonist_action_space = self.environment_class.action_space
        self.protagonist_action_space = self.environment_class.action_space

        self.antagonist_env_instance = None
        self.protagonist_env_instance = None
        self.total_antagonist_reward = 0
        self.total_protagonist_reward = 0
        self.last_antagonist_reward = None
        self.last_protagonist_reward = None

        self.observations, self.rewards, self.dones, self.info = {}, {}, {}, {}
        self.initial_env = None
        self.bit = None
        self.active_antagonist = None
        self.seed = self.builder_obs_space.sample()

        self.reset()

    def reset(self):
        self.observations = {"builder": self.seed}
        self.rewards = {"builder": None}
        self.dones = {agent: False for agent in self.agents}
        self.dones["__all__"] = False
        self.info = {}

        self.initial_env = None
        self.bit = None
        self.active_antagonist = None

        self.total_antagonist_reward = 0
        self.total_protagonist_reward = 0

        return self.observations

    def step(self, action_dict):
        if "builder" in action_dict.keys():
            self.initial_env = action_dict["builder"]
            self.observations = {"describer": (self.initial_env, self.seed)}
            self.rewards = {"describer": None}

        elif "describer" in action_dict.keys():
            self.bit = action_dict["describer"]
            if self.double_antagonist:
                if self.bit == 0:
                    self.active_antagonist = "antagonist_0"
                else:
                    self.active_antagonist = "antagonist_1"
            else:
                self.active_antagonist = "antagonist"

            self.antagonist_env_instance = self.environment_class(self.initial_env)
            self.protagonist_env_instance = self.environment_class(self.initial_env)

            antagonist_obs = self.antagonist_env_instance.reset()
            if not self.double_antagonist:
                antagonist_obs = (antagonist_obs, self.bit)
            protagonist_obs = self.protagonist_env_instance.reset()

            self.observations = {self.active_antagonist: antagonist_obs,
                                 "protagonist": protagonist_obs}
            self.rewards = {self.active_antagonist: None,
                            "protagonist": None}

        else:
            # Perform actions for antagonist and protagonist until they are done.

            self.observations = {}
            self.rewards = {}

            if self.active_antagonist in action_dict.keys():
                [antagonist_obs, antagonist_reward, antagonist_done, _] = \
                    self.antagonist_env_instance.step(action_dict[self.active_antagonist])

                self.total_antagonist_reward += antagonist_reward
                self.dones[self.active_antagonist] = antagonist_done

                if not antagonist_done:
                    if self.double_antagonist:
                        self.observations[self.active_antagonist] = antagonist_obs
                    else:
                        self.observations[self.active_antagonist] = (antagonist_obs, self.bit)
                    self.rewards[self.active_antagonist] = antagonist_reward
                else:
                    self.last_antagonist_reward = antagonist_reward

            if "protagonist" in action_dict.keys():
                [protagonist_obs, protagonist_reward, protagonist_done, _] = \
                    self.protagonist_env_instance.step(action_dict["protagonist"])

                self.total_protagonist_reward += protagonist_reward
                self.dones["protagonist"] = protagonist_done

                if not protagonist_done:
                    self.observations["protagonist"] = protagonist_obs
                    self.rewards["protagonist"] = protagonist_reward
                else:
                    self.last_protagonist_reward = protagonist_reward

            if self.dones[self.active_antagonist] and self.dones["protagonist"]:
                # In this case, both agents are finished and we must
                self.observations = {
                    "builder": self.seed,
                    "describer": (self.initial_env, self.seed),
                    self.active_antagonist: self.antagonist_obs_space.sample(),
                    "protagonist": self.protagonist_obs_space.sample()  # TODO Might be a neater way of doing this
                }

                self.rewards = {
                    "builder": self.total_antagonist_reward - self.total_protagonist_reward,
                    "describer": self.total_antagonist_reward - self.total_protagonist_reward,
                    self.active_antagonist: self.last_antagonist_reward,
                    "protagonist": self.last_protagonist_reward
                }

                self.dones = {agent: True for agent in self.agents}
                self.dones["__all__"] = True

        return [self.observations, self.rewards, self.dones, self.info]


def environment_creator_creator(environment_class, double_antagonist=False):

    def environment_creator(env_config=None):
        env = SeparatorEnv(environment_class, double_antagonist, env_config)
        return env

    return environment_creator
