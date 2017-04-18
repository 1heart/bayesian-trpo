from __future__ import print_function
import numpy as np
import math

import pandas as pd

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


def go(curr_iter, step_size, n_itr=40):
    curr_dir = '/Users/yixin/bayesian-trpo/logs/' + str(curr_iter)

    def run_task(*_):
        env = normalize(CartpoleEnv())

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100,
            n_itr=n_itr,
            discount=0.99,
            step_size=step_size,
            # Uncomment both lines (this and the plot parameter below) to enable plotting
            # plot=True,
        )
        algo.train()

    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # exp_name=str(curr_iter),
        log_dir=curr_dir,
        plot=True,
    )
    df = pd.read_csv(curr_dir + '/progress.csv')
    best_return = df['AverageDiscountedReturn'].max()
    return -1 * best_return


def main(job_id, params):
    return go(job_id, params['step_size'])
