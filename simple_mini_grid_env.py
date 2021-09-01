import numpy as np
from gym.spaces import Discrete, Tuple
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import ray


OLD_TO_NEW = {
    1: 0,  # Empty
    2: 1,  # Wall
    8: 2,  # Goal
}

NEW_ID_TO_STR = {
    0: ' ',
    1: 'W',
    2: 'G',
}


def render_obs(obs):
    for x in obs:
        line = ""
        for y in x:
            line += 2 * NEW_ID_TO_STR[y]
        print(line)


class SimpleMiniGrid(MiniGridEnv):

    size = 11
    obs_size = 7

    env_space = Discrete(2)  # TODO Should really think of a better way of doing than duplicating
    action_space = Discrete(3)
    observation_space = Discrete(3)
    observation_space = Tuple((observation_space,) * obs_size)
    observation_space = Tuple((observation_space,) * obs_size)

    def __init__(self, goal_position):
        self.agent_start_pos = (int((self.size-1)/2), 1)
        self.agent_start_dir = 1

        if goal_position == 0:
            self.goal_position = (1, self.size-2)
        else:
            self.goal_position = (self.size-2, self.size-2)

        super().__init__(grid_size=self.size,
                         max_steps=4 * self.size * self.size,
                         see_through_walls=True)

        self.env_space = Discrete(2)
        self.action_space = Discrete(3)
        self.observation_space = Discrete(3)
        self.observation_space = Tuple((self.observation_space,) * self.obs_size)
        self.observation_space = Tuple((self.observation_space,) * self.obs_size)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the goal position
        (x, y) = self.goal_position
        self.put_obj(Goal(), x, y)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self):
        obs = super().reset()
        obs = obs['image'][:, :, 0]
        obs = [[OLD_TO_NEW[y] for y in x] for x in obs]
        return obs

    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        obs = obs['image'][:, :, 0]
        obs = [[OLD_TO_NEW[y] for y in x] for x in obs]
        return [obs, reward, done, info]

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


def test_roll_out():
    my_env = SimpleMiniGrid(0)
    print(my_env.__str__())
    obs = my_env.reset()
    done = False
    count = 0

    while not done:
        count += 1
        print()
        print("Iteration %s" % count)
        print()
        print("Observation:")
        render_obs(obs)
        print()
        action = my_env.action_space.sample()
        print("Action: %s" % action)
        [obs, reward, done, _] = my_env.step(action)
        print()
        print(my_env.__str__())
        print()
        print("Reward: %s" % reward)
        print("Done: %s" % done)


def env_creator():
    bit = np.random.randint(0, 1)
    return SimpleMiniGrid(bit)


def train_agent():
    register_env("SimpleMiniGrid", SimpleMiniGrid)

    ray.init()

    trainer = ppo.PPOTrainer(env="SimpleMiniGrid")
    for _ in range(10):
        result = trainer.train()
        print(result)

    ray.shutdown()


# test_roll_out()
# train_agent()
