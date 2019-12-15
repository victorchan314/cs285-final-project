import gym
from metaworld.benchmarks import ML10
from rllab.envs.env_spec import EnvSpec
from rllab.envs.gym_env import convert_gym_space

class ML10Wrapper(gym.Env):
    def __init__(self, wrapped_env):
        self.env = wrapped_env
        observation_space = wrapped_env.observation_space
        action_space = wrapped_env.action_space
        self.observation_space = convert_gym_space(observation_space)
        self.action_space = convert_gym_space(action_space)
        #self.env_spec = EnvSpec(self.observation_space, self.action_space)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
