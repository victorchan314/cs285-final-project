# Environment Imports
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
#import dnc.envs as dnc_envs

from rllab.envs.gym_env import GymEnv2
from metaworld.benchmarks import ML1

# Algo Imports

from sandbox.rocky.tf.algos.trpo import TRPO
# from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

# Experiment Imports

from rllab.misc.instrument import stub, run_experiment_lite

def run_task(args,*_):
    
    #env = TfEnv(normalize(dnc_envs.create_stochastic('pick'))) # Cannot be solved easily by TRPO
    metaworld_env = ML1.get_train_tasks('pick-place-v1')  # Create an environment with task `pick_place`
    tasks = metaworld_env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
    metaworld_env.set_task(tasks[0])  # Set task
    # print(metaworld_env.id)
    # print("HERE")
    # import pdb;pdb.set_trace()
    metaworld_env = GymEnv2(metaworld_env)
    # metaworld_env.observation_space = convert_gym_space(metaworld_env.observation_space)
    # metaworld_env.action_space = convert_gym_space(metaworld_env.action_space)


    # env = metaworld_env
    env = TfEnv(normalize(metaworld_env)) # Cannot be solved easily by TRPO

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
        batch_size=10000,
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
