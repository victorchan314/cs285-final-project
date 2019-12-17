# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from metaworld.benchmarks import ML10
from dnc.metaworld.env_wrappers import ML10Wrapper

# Algo Imports

import dnc.algos.trpo as dnc_trpo
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite

def run_task(args,*_):

    metaworld_train_env = ML10.get_train_tasks()
    wrapped_train_env = ML10Wrapper(metaworld_train_env)
    env = TfEnv(wrapped_train_env)
    wrapped_train_env_partitions = wrapped_train_env.get_partitions()
    partitions = [TfEnv(partition) for partition in wrapped_train_env_partitions]

    metaworld_test_env = ML10.get_test_tasks()
    wrapped_test_env = ML10Wrapper(metaworld_test_env)
    test_env = TfEnv(wrapped_test_env)
    
    policy_class = GaussianMLPPolicy
    policy_kwargs = dict(
        min_std=1e-2,
        hidden_sizes=(150, 100, 50),
    )

    baseline_class = LinearFeatureBaseline

    algo = dnc_trpo.TRPO(
        env=env,
        test_env=test_env,
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
    log_dir='data/dnc/ml10',
    n_parallel=12,
    snapshot_mode="last",
    seed=1,
    variant=dict(),
    use_cloudpickle=True,
)
