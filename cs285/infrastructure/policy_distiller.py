import numpy as np

class PolicyDistiller(object):
    def __init__(self, replay_buffer, n_iter=1000, batch_size=1000):
        self.replay_buffer = replay_buffer
        self.n_iter = n_iter
        self.batch_size = batch_size

    def distill_policy(self, old_policy, new_policy):
        for _ in self.n_iter:
            rollouts = self.replay_buffer.sample_random_rollouts(self.batch_size)
            old_policy_actions = old_policy.get_action(rollouts)
            new_policy.update(rollouts, old_policy_actions)
