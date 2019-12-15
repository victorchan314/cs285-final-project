# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import convert_gym_space
#import dnc.envs as dnc_envs

from metaworld.benchmarks import ML1

# Algo Imports

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite

def run_task(args,*_):
    
    #env = TfEnv(normalize(dnc_envs.create_stochastic('pick'))) # Cannot be solved easily by TRPO
    #env = TfEnv(normalize(CartpoleEnv()))
    env = TfEnv(CartpoleEnv())
    #metaworld_env = ML1.get_train_tasks("pick-place-v1")
    #tasks = metaworld_env.sample_tasks(1)
    #metaworld_env.set_task(tasks[0])
    #metaworld_env._observation_space = convert_gym_space(metaworld_env.observation_space)
    #metaworld_env._action_space = convert_gym_space(metaworld_env.action_space)
    #env = TfEnv(normalize(metaworld_env))

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        min_std=1e-2,
        hidden_sizes=(150, 100, 50),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
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


run_experiment_lite(
    run_task,
    log_dir='data/trpo/pick',
    n_parallel=12,
    snapshot_mode="last",
    seed=1,
    variant=dict(),
    use_cloudpickle=True
)
