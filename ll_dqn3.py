import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=200000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0001
        self.tau = .12

        self.model = self.create_model()
        print(self.model)        
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(48, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def report(self):
        return self.epsilon

    def replay(self):
        batch_size = 24
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    env = gym.make("LunarLander-v2")
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500
    max_reward = -float('inf')
    reward_history = []
    dqn_agent = DQN(env=env)

    for trial in range(trials):
        cur_state = env.reset().reshape(1, 8)
        steps, total_reward = 0, 0

        while True:

            env.render()
            action = dqn_agent.act(cur_state)
            steps += 1
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, 8)

            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model
            total_reward += reward
            cur_state = new_state

            if done:
                break

        reward_history.append(total_reward)

        if trial % 10 == 0:
            rh = pd.Series(reward_history)
            rhm = rh.rolling(window=10, min_periods=0).mean()

            plt.scatter(range(len(reward_history)), reward_history)
            plt.plot(range(len(reward_history)), rhm)
            plt.scatter(range(len(reward_history)), reward_history)
            plt.savefig('ll_dqn3.png')

        if total_reward > max_reward and trial > 50:
            max_reward = total_reward
            print("New max reward! {} Trial: {}".format(total_reward, trial))
            dqn_agent.save_model("best_3.model")

        ep = dqn_agent.report()

        print("DQN3 Ended trial {} after {} steps. Reward: {} Epsilon {}".format(trial, steps, total_reward, ep))


if __name__ == "__main__":
    main()