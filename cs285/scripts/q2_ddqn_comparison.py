import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str)
    args = parser.parse_args()

    plt.figure(figsize=(10, 5))

    dirs = sorted(os.listdir(args.logdir))
    dqn_returns = []
    ddqn_returns = []

    for d in dirs:
        parameters = d.split("_")

        if len(parameters) >= 5 and parameters[0] == "dqn" and (parameters[1] == "q2" or parameters[3] == "q2"):
            path = os.path.join(args.logdir, d)
            train_returns = []

            if os.path.isdir(path):
                for file_name in os.listdir(path):
                    if file_name.startswith("events.out.tfevents"):
                        for event in tf.train.summary_iterator(os.path.join(path, file_name)):
                            for v in event.summary.value:
                                if v.tag == "Train_AverageReturn":
                                    train_returns.append(v.simple_value)

            if parameters[1] == "q2":
                algorithm = "DQN"
                seed = int(parameters[3])
                returns = dqn_returns
                color = "red"
            elif parameters[3] == "q2":
                algorithm = "DDQN"
                seed = int(parameters[5])
                returns = ddqn_returns
                color="blue"

            returns.append(np.array(train_returns))
            plt.plot(train_returns, c=color, ls="--", alpha=0.3)

    plt.plot(sum(dqn_returns) / len(dqn_returns), label="Average DQN return", c="red")
    plt.plot(sum(ddqn_returns) / len(ddqn_returns), label="Average DDQN return", c="blue")

    plt.plot()
    plt.title("Lunar Lander Returns for DQN and DDQN")
    plt.xlabel("Iteration")
    plt.ylabel("Train_AverageReturn")
    plt.legend()
    plt.show()
