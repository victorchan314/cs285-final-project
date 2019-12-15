import gym
from metaworld.benchmarks import ML10
from rllab.envs.env_spec import EnvSpec
from rllab.envs.gym_env import convert_gym_space
from rllab.envs.proxy_env import ProxyEnv

class ML10Wrapper(gym.Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        observation_space = wrapped_env.observation_space
        action_space = wrapped_env.action_space
        self.observation_space = convert_gym_space(observation_space)
        self.action_space = convert_gym_space(action_space)
        #self.env_spec = EnvSpec(self.observation_space, self.action_space)

    def step(self, action):
        return self._wrapped_env.step(action)

    def reset(self):
        return self._wrapped_env.reset()

    def render(self, mode="human"):
        return self._wrapped_env.render(mode)

    def close(self):
        return self._wrapped_env.close()

    def seed(self, seed=None):
        return self._wrapped_env.seed(seed)

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass
