import argparse
import datetime as dt

# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from metaworld.benchmarks import ML10, ML45, MT10, MT50
from dnc.metaworld.env_wrappers import MetaworldWrapper

# Algo Imports

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite


benchmark = None


def run_task(args,*_):
    metaworld_train_env = benchmark.get_train_tasks()
    wrapped_train_env = MetaworldWrapper(metaworld_train_env)
    env = TfEnv(wrapped_train_env)

    metaworld_test_env = benchmark.get_test_tasks()
    wrapped_test_env = MetaworldWrapper(metaworld_test_env)
    test_env = TfEnv(wrapped_test_env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        min_std=1e-2,
        hidden_sizes=(150, 100, 50),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        test_env=test_env,
        policy=policy,
        baseline=baseline,
        batch_size=20000,
        # batch_size=100,
        force_batch_sampler=True,
        max_path_length=50,
        discount=1,
        step_size=0.02,
    )
    
    algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", "--b", help="ml10 or ml45")
    args = parser.parse_args()

    if args.benchmark == "ml10":
        benchmark = ML10
    elif args.benchmark == "ml45":
        benchmark = ML45
    elif args.benchmark == "mt10":
        benchmark = MT10
    elif args.benchmark == "mt50":
        benchmark = MT50
    else:
        raise ValueError("Invalid benchmark {}".format(args.benchmark))

run_experiment_lite(
    run_task,
    log_dir='data/trpo/{}_{}'.format(args.benchmark, dt.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
    n_parallel=12,
    snapshot_mode="last",
    seed=1,
    variant=dict(),
    use_cloudpickle=True
)
