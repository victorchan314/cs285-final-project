import numpy as np
from threading import Lock, Thread
import time

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    if 'track' in env.env.model.camera_names:
                        image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # End the rollout if the rollout ended 
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = int(done or steps >= max_path_length)
        terminals.append(rollout_done)
        
        if rollout_done: 
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)

    return paths, timesteps_this_batch

def sample_trajectories_parallelized(envs, policy, min_timesteps_per_batch, max_path_length, num_threads, render=False, render_mode=('rgb_array')):
    thread_pool = EnvSamplingThreadPool(envs, policy, min_timesteps_per_batch, max_path_length, render, render_mode)
    paths, timesteps_this_batch = thread_pool.sample()

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    
    paths = [sample_trajectory(env, policy, max_path_length, render, render_mode) for _ in range(ntraj)]

    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

############################################
############################################

class EnvSamplingThread(Thread):
    def __init__(self, pool, lock, min_timesteps, env, policy, max_path_length, render, render_mode):
        self.pool = pool
        self.lock = lock
        self.min_timesteps = min_timesteps

        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.render = render
        self.render_mode = render_mode
        super(EnvSamplingThread, self).__init__()

    def run(self):
        while self.pool.total_timesteps < self.min_timesteps:
            path = sample_trajectory(self.env, self.policy, self.max_path_length, self.render, self.render_mode)
            path_length = get_pathlength(path)

            with self.lock:
                if self.pool.total_timesteps >= self.min_timesteps:
                    pass
                else:
                    self.pool.total_timesteps += path_length
                    self.pool.paths.append(path)

class EnvSamplingThreadPool(object):
    def __init__(self, envs, policy, min_timesteps, max_path_length, render, render_mode):
        self.lock = Lock()
        self.threads = [EnvSamplingThread(self, self.lock, min_timesteps, env, policy, max_path_length, render, render_mode) for env in envs]
        self.paths = []
        self.total_timesteps = 0

    def sample(self):
        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()

        return self.paths, self.total_timesteps
