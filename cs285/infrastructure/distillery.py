import numpy as np

class Distillery(object):
    def __init__(self, replay_buffer, n_iter=1000, batch_size=1000):
        self.replay_buffer = replay_buffer
        self.n_iter = n_iter
        self.batch_size = batch_size

    def distill_policy(self, teacher, student, estimate_advantage=None):
        loss = 0
        for _ in range(self.n_iter):
            obs, acs, rews, next_obs, dones = self.replay_buffer.sample_random_data(self.batch_size)
            teacher_actions = teacher.get_action(obs)
            advantage = estimate_advantage(obs, next_obs, rews, dones)
            loss = student.update(obs, teacher_actions, adv_n=advantage)

        return loss
