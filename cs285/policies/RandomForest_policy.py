import numpy as np
import tensorflow as tf
from .base_policy import BasePolicy
from cs285.infrastructure.tf_utils import build_mlp
import tensorflow_probability as tfp
from tensorflow.contrib.tensor_forest.python import tensor_forest

class RandomForestPolicy(BasePolicy):

    def __init__(self,
        sess,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        training=True,
        policy_scope='policy_vars',
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.sess = sess
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # build TF graph
        with tf.variable_scope(policy_scope, reuse=tf.AUTO_REUSE):
            self.build_graph()

        # saver for policy variables that are not related to training
        self.policy_vars = [v for v in tf.all_variables() if policy_scope in v.name and 'train' not in v.name]
        self.policy_saver = tf.train.Saver(self.policy_vars, max_to_keep=None)

        self.forest_params = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

    ##################################

    def build_graph(self):
        self.forest_graph = tensor_forest.RandomForestGraphs(hparams)
        if self.training:
            with tf.variable_scope('train', reuse=tf.AUTO_REUSE):
                self.define_train_op()

    ##################################

    def define_placeholders(self):
        # placeholder for observations
        self.observations_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        # placeholder for actions
        if self.discrete:
            self.actions_pl = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.actions_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)


    def build_action_sampling(self):
        if self.discrete:
            logits_na = self.parameters
            self.sample_ac = tf.squeeze(tf.multinomial(logits_na, num_samples=1), axis=1)
        else:
            mean, logstd = self.parameters
            self.sample_ac = mean + tf.exp(logstd) * tf.random_normal(tf.shape(mean), 0, 1)

    def define_train_op(self):
        # Get training graph and loss
        train_op = forest_graph.training_graph(self.observations_pl, self.actions_pl)
        loss_op = forest_graph.training_loss(self.observation_pl, self.actions_pl)


    def define_log_prob(self):
        if self.discrete:
            #log probability under a categorical distribution
            logits_na = self.parameters
            self.logprob_n = tf.distributions.Categorical(logits=logits_na).log_prob(self.actions_pl)
        else:
            #log probability under a multivariate gaussian
            mean, logstd = self.parameters
            self.logprob_n = tfp.distributions.MultivariateNormalDiag(
                loc=mean, scale_diag=tf.exp(logstd)).log_prob(self.actions_pl)

    def build_baseline_forward_pass(self):
        self.baseline_prediction = tf.squeeze(build_mlp(self.observations_pl, output_size=1, scope='nn_baseline', n_layers=self.n_layers, size=self.size))

    ##################################

    def save(self, filepath):
        self.policy_saver.save(self.sess, filepath, write_meta_graph=False)

    def restore(self, filepath):
        self.policy_saver.restore(self.sess, filepath)

    ##################################


    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None, qvals=None):
        assert(self.training, 'Policy must be created with training=True in order to perform training updates...')

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.observations_pl: observations, self.actions_pl: acs_na, self.adv_n: adv_n})

        if self.nn_baseline:
            targets_n = (qvals - np.mean(qvals))/(np.std(qvals)+1e-8)
            # TODO: update the nn baseline with the targets_n
            # HINT1: run an op that you built in define_train_op
            self.sess.run(self.baseline_update_op, feed_dict={self.observations_pl: observations, self.targets_n: targets_n})
        return loss

    # query the neural net that's our 'policy' function, as defined by an mlp above
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):

        # TODO: GETTHIS from HW1
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        # HINT1: you will need to call self.sess.run
        # HINT2: the tensor we're interested in evaluating is self.sample_ac
        # HINT3: in order to run self.sample_ac, it will need observation fed into the feed_dict
        return self.sess.run(self.sample_ac, feed_dict={self.observations_pl : observation})

    