from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger

from dnc.sampler.policy_sampler import Sampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

import joblib
import tensorflow as tf
import numpy as np
import time


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc
    """

    def __init__(
            self,
            # DnC Options
            env,
            partitions,
            policy_class,
            policy_kwargs=dict(),
            params_pkl=None,
            baseline_class=LinearFeatureBaseline,
            baseline_kwargs=dict(),
            distillation_period=50,
            # Base BatchPolopt Options
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,  # Batch size for each partition
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            force_batch_sampler=False,
            test_env=None,
            save_data=False,
            should_optimize_policy=True,
            **kwargs
    ):
        """
        DnC options

        :param env: Central environment trying to solve
        :param partitions: A list of environments to use as partitions for central environment
        :param policy_class: The policy class to use for global and local policies (for example GaussianMLPPolicy from sandbox.rocky.tf.policies.gaussian_mlp_policy)
        :param policy_kwargs: A dictionary of additional parameters used for policy (beyond name and env_spec)
        :param baseline_class: The baseline class used for local policies (for example LinearFeatureBaseline from rllab.baselines.linear_feature_baseline)
        :param baseline_kwargs: A dictionary of additional parameters used for baselien (beyond env_spec)
        :param distillation_period: How often to distill local policies into global policy, and reset

        Base RLLAB options

        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.params_pkl = params_pkl

        if not self.params_pkl is None:
            def get_numbered_order(d, name):
                l = [d[name]]

                sorted_keys = sorted([k for k in d.keys() if k.startswith(name) and len(k) > len(name)])

                for k in sorted_keys:
                    l.append(d[k])

                return l

            sess = tf.InteractiveSession()
            params = joblib.load(self.params_pkl)
            sess.close()

            policy = get_numbered_order(params, "policy")
            self.policy = policy[0]
            self.local_policies = policy[1:]
        else:
            self.policy = policy_class(name='central_policy', env_spec=env.spec, **policy_kwargs)
            self.local_policies = [
                policy_class(name='local_policy_%d' % (n), env_spec=env.spec, **policy_kwargs) for n in range(len(partitions))
            ]

        self.env = env
        self.env_partitions = partitions
        self.n_parts = len(self.env_partitions)

        self.baseline = baseline_class(env_spec=env.spec, **baseline_kwargs)

        self.local_baselines = [
            baseline_class(env_spec=env.spec, **baseline_kwargs) for n in range(self.n_parts)
        ]

        self.distillation_period = distillation_period

        # Housekeeping

        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon

        self.local_samplers = [
            Sampler(
                env=env,
                policy=policy,
                baseline=baseline,
                scope=scope,
                batch_size=batch_size,
                max_path_length=max_path_length,
                discount=discount,
                gae_lambda=gae_lambda,
                center_adv=center_adv,
                positive_adv=positive_adv,
                whole_paths=whole_paths,
                fixed_horizon=fixed_horizon,
                force_batch_sampler=force_batch_sampler
            ) for env, policy, baseline in zip(self.env_partitions, self.local_policies, self.local_baselines)
        ]

        self.global_sampler = Sampler(
                env=self.env,
                policy=self.policy,
                baseline=self.baseline,
                scope=scope,
                batch_size=batch_size,
                max_path_length=max_path_length,
                discount=discount,
                gae_lambda=gae_lambda,
                center_adv=center_adv,
                positive_adv=positive_adv,
                whole_paths=whole_paths,
                fixed_horizon=fixed_horizon,
                force_batch_sampler=force_batch_sampler
        )

        self.test_env = test_env
        self.save_data = save_data
        self.should_optimize_policy = should_optimize_policy

        if not self.test_env is None:
            self.test_sampler = Sampler(
                    env=self.test_env,
                    policy=self.policy,
                    baseline=self.baseline,
                    scope=scope,
                    batch_size=batch_size,
                    max_path_length=max_path_length,
                    discount=discount,
                    gae_lambda=gae_lambda,
                    center_adv=center_adv,
                    positive_adv=positive_adv,
                    whole_paths=whole_paths,
                    fixed_horizon=fixed_horizon,
                    force_batch_sampler=force_batch_sampler
            )

        self.init_opt()

    def start_worker(self):
        for sampler in self.local_samplers:
            sampler.start_worker()
        self.global_sampler.start_worker()

    def shutdown_worker(self):
        for sampler in self.local_samplers:
            sampler.shutdown_worker()
        self.global_sampler.shutdown_worker()

    def train(self, sess=None):
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            sess.__enter__()
            sess.run(tf.initialize_all_variables())
        else:
            sess.run(tf.initialize_variables(list(tf.get_variable(name) for name in sess.run(tf.report_uninitialized_variables()))))

        self.start_worker()
        start_time = time.time()

        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()

            with logger.prefix('itr #%d | ' % itr):
                all_paths = []
                logger.log("Obtaining samples...")
                for sampler in self.local_samplers:
                    all_paths.append(sampler.obtain_samples(itr))

                logger.log("Processing samples...")
                all_samples_data = []
                for n, (sampler, paths) in enumerate(zip(self.local_samplers, all_paths)):
                    with logger.tabular_prefix(str(n)):
                        all_samples_data.append(sampler.process_samples(itr, paths))

                logger.log("Logging diagnostics...")
                self.log_diagnostics(all_paths,)

                if self.should_optimize_policy:
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, all_samples_data)

                if not self.test_env is None:
                    logger.log("Obtaining test samples...")
                    test_paths = self.test_sampler.obtain_samples(itr)
                    with logger.tabular_prefix("Test"):
                        test_samples = self.test_sampler.process_samples(itr, test_paths)
                        logger.record_tabular("TestSuccessRate", np.mean(test_samples["env_infos"]["success"]))

                successes = 0.0
                trials = 0.0
                for i, samples_data in enumerate(all_samples_data):
                    success = samples_data["env_infos"]["success"]
                    logger.record_tabular("SuccessRate{}".format(i), np.mean(success))
                    successes += np.sum(success)
                    trials += success.shape[0]

                success_rate = successes / trials
                logger.record_tabular("SuccessRate", success_rate)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr, all_samples_data)  # , **kwargs)
                logger.save_itr_params(itr, params)

                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()

    def log_diagnostics(self, all_paths):
        for n, (env, policy, baseline, paths) in enumerate(zip(self.env_partitions, self.local_policies, self.local_baselines, all_paths)):
            with logger.tabular_prefix(str(n)):
                env.log_diagnostics(paths)
                policy.log_diagnostics(paths)
                baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError()

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """

        d = dict()
        for n, (policy, env) in enumerate(zip(self.local_policies, self.env_partitions)):
            d['policy%d' % n] = policy
            d['env%d' % n] = env

        d['policy'] = self.policy
        d['env'] = self.env

        if self.save_data:
            d["samples_data"] = samples_data

        return d
 
    def optimize_policy(self, itr, samples_data):
        """
        Runs the optimization procedure
        """
        raise NotImplementedError()
