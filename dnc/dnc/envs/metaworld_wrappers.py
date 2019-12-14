from metaworld.benchmarks import ML10
from rllab.envs.env_spec import EnvSpec

clss ML10Wrapper(gym.Env):
    def __init__(self, wrapped_env):
        self.env = wrapped_env
        self.env_spec = EnvSpec(self.env.observation_space, self.env.action_space)

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
