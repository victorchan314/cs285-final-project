import gym
from rllab.envs.env_spec import EnvSpec
from rllab.envs.gym_env import convert_gym_space

class MetaworldWrapper(gym.Env):
    def __init__(self, wrapped_env):
        self.__wrapped_env = wrapped_env
        observation_space = wrapped_env.observation_space
        action_space = wrapped_env.action_space
        self.observation_space = convert_gym_space(observation_space)
        self.action_space = convert_gym_space(action_space)
        self.spec = EnvSpec(self.observation_space, self.action_space)

    def step(self, action):
        return self.__wrapped_env.step(action)

    def reset(self):
        return self.__wrapped_env.reset()

    def render(self, mode="human"):
        return self.__wrapped_env.render(mode)

    def close(self):
        return self.__wrapped_env.close()

    def terminate(self):
        return self.close()

    def seed(self, seed=None):
        return self.__wrapped_env.seed(seed)

    def log_diagnostics(self, paths):
        self.__wrapped_env.log_diagnostics(paths, "MetaworldWrapper")

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def get_partitions(self):
        # For now just using the meta-train tasks as partitions
        # maybe in the future we can use variations too
        partitions = [MetaworldWrapper(x) for x in self.__wrapped_env._task_envs]

        return partitions
