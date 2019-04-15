import gym
import numpy as np
import keras

import matplotlib.pyplot as plt
import pandas as pd


class DQN:
    def __init__(self, env):
        self.env = env
        self.model = keras.models.load_model("./models/best/best_3.model")

    def act(self, state):
        return np.argmax(self.model.predict(state.reshape(1, 8)))


def main():
    env = gym.make("LunarLander-v2")
    trials = 100
    reward_history = []
    agent = DQN(env=env)

    for trial in range(trials):

        ss = env.reset().reshape(1, 8)
        steps, total_reward = 0, 0

        while True:
            env.render()
            action = agent.act(ss)
            steps += 1
            ns, rr, done, _ = env.step(action)
            total_reward += rr
            ss = ns

            if done:
                break

        reward_history.append(total_reward)

    rh = pd.Series(reward_history)
    print("Average reward after 100 trials: {} ".format(rh.mean()))
    plt.figure(figsize=(14, 7))
    plt.plot(rh)
    plt.scatter(rh.index, reward_history)
    title_string = "Test Results | Mean Score: {}".format(rh.mean())
    plt.title(title_string)
    plt.savefig('./plots/{}_test_results.png'.format('best_model'))


if __name__ == "__main__":
    main()
