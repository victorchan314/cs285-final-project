# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
import dnc.envs as dnc_envs
from metaworld.benchmarks import ML1

env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
env.set_task(tasks[0])  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action


from metaworld.benchmarks import ML10
ml10_train_env = ML10.get_train_tasks()

# Algo Imports

import dnc.algos.trpo as dnc_trpo
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite

def run_task(args,*_):

    base_env = dnc_envs.create_stochastic('pick')
    base_partitions = dnc_envs.create_env_partitions(base_env, k=4)

    env = TfEnv(normalize(base_env))
    partitions = [TfEnv(normalize(part_env)) for part_env in base_partitions]
    
    policy_class = GaussianMLPPolicy
    policy_kwargs = dict(
        min_std=1e-2,
        hidden_sizes=(150, 100, 50),
    )

    baseline_class = LinearFeatureBaseline

    algo = dnc_trpo.TRPO(
        env=env,
        partitions=partitions,
        policy_class=policy_class,
        policy_kwargs=policy_kwargs,
        baseline_class=baseline_class,
        batch_size=20000,
        n_itr=500,
        force_batch_sampler=True,
        max_path_length=50,
        discount=1,
        step_size=0.02,
    )
    
    algo.train()

run_experiment_lite(
    run_task,
    log_dir='data/dnc/pick',
    n_parallel=12,
    snapshot_mode="last",
    seed=1,
    variant=dict(),
    use_cloudpickle=True,
)
