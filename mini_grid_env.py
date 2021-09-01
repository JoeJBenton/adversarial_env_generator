from gym.spaces import Discrete, Tuple
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Wall
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import ray
import numpy as np

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


class MiniGridGeneral(MiniGridEnv):
    size = 11
    obs_size = 7

    env_space = Discrete(48)
    action_space = Discrete(3)
    observation_space = Discrete(3)
    observation_space = Tuple((observation_space,) * obs_size)
    observation_space = Tuple((observation_space,) * obs_size)

    def __init__(self, goal_position):
        self.agent_start_pos = (int((self.size - 1) / 2), 1)
        self.agent_start_dir = 1

        goal_x = (goal_position % 8) + 1
        goal_y = self.size - 2 - (goal_position // 8)
        if goal_x > 4:
            goal_x += 1
        self.goal_position = (goal_x, goal_y)

        # TODO Remove
        self.count = 0
        self.display = []

        super().__init__(grid_size=self.size,
                         max_steps=4 * self.size * self.size,
                         see_through_walls=True)

        self.env_space = Discrete(48)
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
        # print("Goal pos " + str((x, y)))
        # input()

        # Place T shape of walls
        x = int((self.size - 1) / 2)
        for y in range(3, self.size - 1):
            self.put_obj(Wall(), x, y)

        y = 3
        for x in range(2, self.size - 2):
            self.put_obj(Wall(), x, y)

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

        # TODO Remove
        self.count = 0
        self.display = []

        return obs

    def step(self, action):
        (obs, reward, done, info) = super().step(action)
        obs = obs['image'][:, :, 0]
        obs = [[OLD_TO_NEW[y] for y in x] for x in obs]

        # TODO Remove this
        # self.count += 1
        # if self.count % 10 == 0:
        #     self.display.append(self.__str__())
        # if done:
        #     print("Count: %s" % self.count)
        #     for string in self.display:
        #         print(string)

        return [obs, reward, done, info]

    def partition_function(self, env):
        if (env % 8) + 1 > 4:
            return "R"
        else:
            return "L"

    def partition_keys(self):
        return ["L", "R"]

    def optimal_partition(self, obs):
        for env in range(48):
            if (env % 8) + 1 > 4 and obs[env] == 1:
                return 1

        return 0


class MiniGridRandomGoal(MiniGridGeneral):

    def __init__(self, config=None):
        goal_position = Discrete(48).sample()
        super().__init__(goal_position)

    def reset(self):
        goal_position = Discrete(48).sample()
        goal_x = (goal_position % 8) + 1
        goal_y = self.size - 2 - (goal_position // 8)
        if goal_x > 4:
            goal_x += 1
        self.goal_position = (goal_x, goal_y)

        return super().reset()


def test_roll_out():
    my_env = MiniGridRandomGoal()
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


def compute_roll_outs(trainer, iterations, display=False):
    if display:
        print("DISPLAYING ROLL OUTS...\n")

    av_roll_out_length = 0

    for count in range(iterations):
        if display:
            print("Roll out %s" % (count + 1))
        env_instance = MiniGridRandomGoal()
        obs = env_instance.reset()
        done = False
        roll_out_length = 0

        while not done:
            roll_out_length += 1
            action = trainer.compute_action(obs)
            [obs, reward, done, _] = env_instance.step(action)
            if display:
                print(env_instance.__str__())
                print()

        if display:
            print("Roll out length: %s\n" % roll_out_length)

        av_roll_out_length += roll_out_length

    av_roll_out_length = av_roll_out_length / iterations
    return av_roll_out_length


def train_agent():
    register_env("MiniGridGeneral", MiniGridRandomGoal)

    ray.init()

    trainer = ppo.PPOTrainer(env="MiniGridGeneral", config={
        "train_batch_size": 4000,
        "lr": 1e-4,
        "gamma": 0.999,
        "model": {
            "use_lstm": False
        },
    })

    result = trainer.train()
    count = 0

    while result["episode_reward_min"] < 80 and count < 80:
        result = trainer.train()
        av_roll_out_length = compute_roll_outs(trainer, 10)
        print(result)
        print("Average rollout length: %s" % av_roll_out_length)
        # input()
        count += 1

    compute_roll_outs(trainer, 5, True)

    ray.shutdown()


if __name__ == "__main__":
    # test_roll_out()
    train_agent()
