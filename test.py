import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from sacred import Experiment
from sacred.observers import FileStorageObserver
from separator_wrapper import environment_creator_creator
from test_policies import UniformBuilder, perfect_describer_creator
from single_bit_env import SingleBit
from multi_bit_env import MultiBit
from sequential_multi_bit_env import SequentialMultiBit
from simple_mini_grid_env import SimpleMiniGrid
from mini_grid_env import MiniGridGeneral
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from os import getcwd


env_string_to_env_class = {
    "single_bit": SingleBit,
    "multi_bit": MultiBit,
    "sequential_multi_bit": SequentialMultiBit,
    "simple_mini_grid": SimpleMiniGrid,
    "mini_grid": MiniGridGeneral
}


def test_roll_out(env_creator):
    """
    Produces and displays one test roll out of the separator environment
    :param env_creator: Environment creator for the separator environment
    :return: None
    """

    my_env = env_creator()

    builder_action_space = my_env.builder_action_space
    describer_action_space = my_env.describer_action_space
    antagonist_action_space = my_env.antagonist_action_space
    protagonist_action_space = my_env.protagonist_action_space

    builder_action = builder_action_space.sample()
    print("Builder env: " + str(builder_action))
    [obs, _, _, _] = my_env.step({"builder": builder_action})
    print("Observations: %s" % obs)

    describer_action = describer_action_space.sample()
    print("Describer bit: %s" % describer_action)
    [obs, _, dones, _] = my_env.step({"describer": describer_action})
    print("Observations: %s" % obs)

    while not dones["__all__"]:
        antagonist_action = antagonist_action_space.sample()
        protagonist_action = protagonist_action_space.sample()

        actions = {}
        if not dones[my_env.active_antagonist]:
            actions[my_env.active_antagonist] = antagonist_action
        if not dones["protagonist"]:
            actions["protagonist"] = protagonist_action
        print("Actions: %s" % actions)
        [obs, rewards, dones, _] = my_env.step(actions)
        print("Observations: %s" % obs)

        print("Rewards: %s" % rewards)


def compute_roll_outs(trainer, env_creator, iterations, display=False):
    """
    Produces a list of roll outs
    :param trainer: Trainer used to compute the roll outs
    :param env_creator: Environment creator for the separator environment
    :param iterations: Number of roll outs to compute
    :param display: Whether to also print the roll outs
    :return: List of roll outs, with each in the form:
    (env, bit, [list of action dicts for antagonist, protagonist])
    """
    # Returns a list of roll_outs, with each in the form:
    #

    roll_outs = []

    if display:
        print("DISPLAYING ROLL OUTS...\n")
    for count in range(iterations):
        if display:
            print("Roll out %s" % (count + 1))

        sep_env_instance = env_creator()
        obs = sep_env_instance.reset()
        # print("Observations: %s" % obs)

        if "builder" in trainer.config["multiagent"]["policies_to_train"]:
            builder_action = trainer.compute_action(obs["builder"], policy_id="builder")
        else:
            processed_builder_obs = trainer.workers.local_worker().preprocessors["builder"].transform(obs["builder"])
            builder_action = trainer.get_policy("builder").compute_actions([processed_builder_obs])[0][0]

        if display:
            print("Builder environment: " + str(builder_action))
        [obs, _, _, _] = sep_env_instance.step({"builder": builder_action})
        # print("Observations: %s" % obs)

        if "describer" in trainer.config["multiagent"]["policies_to_train"]:
            describer_action = trainer.compute_action(obs["describer"], policy_id="describer")
        else:
            processed_describer_obs = \
                trainer.workers.local_worker().preprocessors["describer"].transform(obs["describer"])
            describer_action = trainer.get_policy("describer").compute_actions([processed_describer_obs])[0][0]

        if display:
            print("Describer bit: %s" % describer_action)
        [obs, _, dones, _] = sep_env_instance.step({"describer": describer_action})

        active_antagonist = sep_env_instance.active_antagonist

        action_list = []
        while not dones["__all__"]:
            if display:
                print("Observations: %s" % obs)
                print(sep_env_instance.antagonist_env_instance.__str__())
            actions = {}
            if not dones[active_antagonist]:
                antagonist_action = trainer.compute_action(obs[active_antagonist], policy_id=active_antagonist)
                actions[active_antagonist] = antagonist_action
            if not dones["protagonist"]:
                protagonist_action = trainer.compute_action(obs["protagonist"], policy_id="protagonist")
                actions["protagonist"] = protagonist_action

            [obs, rewards, dones, _] = sep_env_instance.step(actions)
            if display:
                print("Actions: %s" % actions)
                print("Rewards: %s" % rewards)

            action_list.append(actions)

        roll_outs.append((builder_action, describer_action, action_list))

        if display:
            print("")

    return roll_outs


def trainer_creator(environment_class, env_creator, uniform_builder=False, perfect_describer=False):
    """
    Creates the RLlib trainer for the separator environment
    :param environment_class: Class of environments in which the antagonist/protagonist act
    :param env_creator: Environment creator for the separator environment
    :param uniform_builder: Whether to fix the builder policy to be uniform
    :param perfect_describer: Whether to fix the describer policy
    :return: RLlib trainer for the separator environment
    """

    sep_env = env_creator()  # TODO Probably a better way of doing this

    if uniform_builder:
        builder_policy = (UniformBuilder, sep_env.builder_obs_space, sep_env.builder_action_space, {})
    else:
        builder_policy = (None, sep_env.builder_obs_space, sep_env.builder_action_space,
                          {"training_batch_size": 1000,
                           "lr": 5e-7})

    if perfect_describer:
        PerfectDescriber = perfect_describer_creator(environment_class.optimal_partition, environment_class)
        describer_policy = (PerfectDescriber, sep_env.describer_obs_space, sep_env.describer_action_space, {})
    else:
        describer_policy = (None, sep_env.describer_obs_space, sep_env.describer_action_space,
                            {"training_batch_size": 1000,
                             "lr": 1e-4})

    antagonist_policy = (None, sep_env.antagonist_obs_space, sep_env.antagonist_action_space,
                         {"training_batch_size": 1000,
                          "lr": 1e-4,
                          "gamma": 0.999})
    protagonist_policy = (None, sep_env.protagonist_obs_space, sep_env.protagonist_action_space,
                          {"training_batch_size": 1000,
                           "lr": 1e-4,
                           "gamma": 0.999})

    policies_to_train = ["protagonist"]
    if not uniform_builder:
        policies_to_train.append("builder")
    if not perfect_describer:
        policies_to_train.append("describer")
    if sep_env.double_antagonist:
        policies_to_train.append("antagonist_0")
        policies_to_train.append("antagonist_1")
    else:
        policies_to_train.append("antagonist")

    policies = {
        "builder": builder_policy,
        "describer": describer_policy,
        "protagonist": protagonist_policy,
    }
    if sep_env.double_antagonist:
        policies["antagonist_0"] = antagonist_policy
        policies["antagonist_1"] = antagonist_policy
    else:
        policies["antagonist"] = antagonist_policy

    trainer = ppo.PPOTrainer(env="separator_env", config={
        "multiagent": {
            "policies_to_train": policies_to_train,
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    })

    return trainer


def reload_checkpoint(environment_class, env_creator, run_address, iteration, uniform_builder=False,
                      perfect_describer=False):
    """
    Reloads a previous trainer
    :param environment_class: Class of environments in which the antagonist/protagonist act
    :param env_creator: Environment creator for the separator environment
    :param run_address: Address of checkpoint to be loaded
    :param iteration: Iteration to be loaded
    :param uniform_builder: Whether the loaded trainer has a fixed uniform builder policy
    :param perfect_describer: Whether the loaded trainer has a fixed perfect describer policy
    :return: Loaded RLlib trainer
    """

    checkpoint_dir = getcwd() + '/checkpoints/' + run_address + "/"
    checkpoint = checkpoint_dir + "checkpoint_" + str(iteration) + "/checkpoint-" + str(iteration)

    trainer = trainer_creator(environment_class, env_creator, uniform_builder=uniform_builder,
                              perfect_describer=perfect_describer)
    trainer.restore(checkpoint)
    return trainer


def create_logs(trainer, environment_class, env_creator, writer, result, epoch, sample_size):
    """
    Writes logging metrics to writer
    :param trainer: RLlib trainer for the separator environment
    :param environment_class: Class of environments in which the antagonist/protagonist act
    :param env_creator: Environment creator for the separator environment
    :param writer: Log writer object
    :param result: Result dict returned by last iteration of training
    :param epoch: Epoch number
    :param sample_size: Number of roll outs used to produce metrics
    :return: None
    """

    print("Rolling out...")
    roll_outs = compute_roll_outs(trainer, env_creator, sample_size, False)
    print("Done")

    partition_keys = environment_class.partition_keys(environment_class)
    partition_function = environment_class.partition_function

    def builder_logs():
        environments = [env for (env, _, _) in roll_outs]
        bits = [bit for (_, bit, _) in roll_outs]

        # Calculate the environment distribution, partitioning the environments according to partition_function
        env_numbers = {key: 0 for key in partition_keys}
        for env in environments:
            env_numbers[partition_function(environment_class, env)] += 1

        env_distribution = {key: float(env_numbers[key]) / sample_size for key in partition_keys}
        writer.add_scalars("Environment Distribution", env_distribution, epoch)

        # Calculate the bit distribution for each subset of the environment class
        env_numbers_bit_1 = {key: 0 for key in partition_keys}
        for (env, bit) in zip(environments, bits):
            if bit == 1:
                env_numbers_bit_1[partition_function(environment_class, env)] += 1
        bit_distribution = {key: float(env_numbers_bit_1[key]) / max(env_numbers[key], 1) for key in partition_keys}
        writer.add_scalars("Bit Distribution", bit_distribution, epoch)

    def describer_logs():
        pass

    def antagonist_logs():
        pass

    def protagonist_logs():
        pass

    builder_logs()
    describer_logs()
    antagonist_logs()
    protagonist_logs()

    writer.add_scalars("Reward Mean", result["policy_reward_mean"], epoch)

    writer.flush()


def train_in_environment(trainer, environment_class, env_creator, iterations, sample_size):
    """
    Trains the RLlib trainer
    :param trainer: RLlib trainer for the separator environment
    :param environment_class: Class of environments in which the antagonist/protagonist act
    :param env_creator: Environment creator for the separator environment
    :param iterations: Number of epochs
    :param sample_size: Number of roll outs used to produce metrics
    :return: None
    """

    # Set up logging directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = getcwd() + '/logs/' + current_time + '/'
    writer = SummaryWriter(log_dir)

    # Set up checkpoint directory
    checkpoint_dir = getcwd() + '/checkpoints/' + current_time + '/'

    for iteration in range(iterations):
        result = trainer.train()
        print(result["policy_reward_mean"])

        checkpoint = trainer.save(checkpoint_dir)
        print(checkpoint)

        # compute_roll_outs(trainer, 3, True)
        create_logs(trainer, environment_class, env_creator, writer, result, iteration, sample_size)

    writer.close()

    return


ex = Experiment()
ex.observers.append(FileStorageObserver('my_runs'))


@ex.config
def basic_config():
    environment_class = "single_bit"
    double_antagonist = False
    test_policies = {
        "uniform_builder": False,
        "perfect_describer": False,
    }
    start_from_checkpoint = None
    iterations = 50
    sample_size = 20


@ex.named_config
def reload_checkpoint_config():
    environment_class = "mini_grid"
    double_antagonist = True
    test_policies = {
        "uniform_builder": True,
        "perfect_describer": True,
    }
    start_from_checkpoint = {
        "checkpoint_address": "20210824-071932",
        "iteration_number": 50,
    }
    iterations = 50
    sample_size = 20


@ex.named_config
def uniform_builder_config():
    environment_class = "single_bit"
    double_antagonist = False
    test_policies = {
        "uniform_builder": True,
        "perfect_describer": False,
    }
    start_from_checkpoint = None
    iterations = 50
    sample_size = 20


@ex.named_config
def all_test_policies():
    environment_class = "single_bit"
    double_antagonist = True
    test_policies = {
        "uniform_builder": True,
        "perfect_describer": True,
    }
    start_from_checkpoint = None
    iterations = 50
    sample_size = 20


@ex.automain
def my_main(environment_class, double_antagonist, test_policies, start_from_checkpoint, iterations, sample_size):

    environment_class = env_string_to_env_class[environment_class]
    env_creator = environment_creator_creator(environment_class, double_antagonist=double_antagonist)
    register_env("separator_env", env_creator)

    # test_roll_out()

    ray.init()

    if start_from_checkpoint is None:
        my_trainer = trainer_creator(environment_class, env_creator, uniform_builder=test_policies["uniform_builder"],
                                     perfect_describer=test_policies["perfect_describer"])
    else:
        my_trainer = reload_checkpoint(environment_class, env_creator, start_from_checkpoint["checkpoint_address"],
                                       start_from_checkpoint["iteration_number"],
                                       uniform_builder=test_policies["uniform_builder"],
                                       perfect_describer=test_policies["perfect_describer"])

    train_in_environment(my_trainer, environment_class, env_creator, iterations, sample_size)

    compute_roll_outs(my_trainer, env_creator, 20, True)

    ray.shutdown()
