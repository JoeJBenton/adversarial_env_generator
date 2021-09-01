from ray.rllib.policy.policy import Policy


class UniformBuilder(Policy):
    """
    The UniformBuilder policy always selects an environment to build uniformly at random from the environment class.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(
            self,
            obs_batch,
            state_batches=None,
            prev_action_batch=None,
            prev_reward_batch=None,
            info_batch=None,
            episodes=None,
            **kwargs):

        return [self.action_space.sample() for obs in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def perfect_describer_creator(optimal_partition, environment_class):
    """
    Creates a PerfectDescriber class with the given partition function for the given environment class.
    :param optimal_partition: Function partitioning the environment class into two parts that determines how the
        describer policy will act.
    :param environment_class: Class of environments in which the antagonist/protagonist act
    :return: PerfectDescriber policy
    """

    class PerfectDescriber(Policy):
        """"
        The PerfectDescriber policy observes an environment and always communicates the bit given by the optimal
        partition function applied to that bit.
        To customise the describer policy, edit the environment class's optimal partition function.
        """

        def __init__(self, observation_space, action_space, config):
            Policy.__init__(self, observation_space, action_space, config)

        def compute_actions(
                self,
                obs_batch,
                state_batches=None,
                prev_action_batch=None,
                prev_reward_batch=None,
                info_batch=None,
                episodes=None,
                **kwargs):

            return [optimal_partition(environment_class, obs) for obs in obs_batch], [], {}

        def learn_on_batch(self, samples):
            pass

        def get_weights(self):
            pass

        def set_weights(self, weights):
            pass

    return PerfectDescriber
