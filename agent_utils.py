import logging
import random
import numpy as np
import uuid
import gym
from keras.models import Sequential, clone_model, load_model
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DNN:
    def __init__(self, input_shape, output_shape, layers, lr):
        self.connections = "Fully Connected"
        self.model = Sequential()
        self.learning_rate = lr
        self.layers = layers

        for i, layer in enumerate(self.layers):
            size, activation = layer
            if i == 0:
                self.model.add(Dense(size, input_dim=input_shape, activation=activation))
            elif i == len(layers)-1:
                self.model.add(Dense(output_shape))
            else:
               self.model.add(Dense(size, input_dim=input_shape, activation=activation))

    def compile(self):
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return self.model


class Memory:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def record(self, observation):
        ss, aa, rr, ns, done = observation
        self.memory.append([ss, aa, rr, ns, done])

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


class DQNAgent:
    def __init__(self, env, model_architecture, alpha, gamma, ep_decay, ep_min, batch_size, warmup, tau, memory_size, memory_interval=1, train_interval=1, epsilon=1., decay_type='exponential'):

        if warmup < batch_size:
            logging.warning("Provided warmup interval {} is less than batch size {}.".format(warmup, batch_size))
            warmup = batch_size

        # User Provided
        self.env = env
        self.model_architecture = model_architecture
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = ep_decay
        self.epsilon_min = ep_min
        self.batch_size = batch_size
        self.warmup = warmup
        self.tau = tau
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.decay_type = decay_type

        # Derived
        self.obs_dims = self.env.observation_space.shape[0]
        self.act_dims = self.env.action_space.n

        # Instantiate new models and memory
        model = DNN(self.obs_dims, self.act_dims, self.model_architecture, self.alpha)
        self.model = model.compile()
        self.target_model = clone_model(self.model)
        self.memory = Memory(memory_size)

        # Stats
        self.step = 0

    def set_model(self, fn):
        self.model = load_model(fn)

    def get_param_string(self, linebreak=False):
        model_shape = "-".join([str(size) for size, _ in self.model_architecture])
        param_parts = ["lr: {}", "gamma: {}", "ep_decay: {}", "ep_min: {}", "architecture: ({})", "tau: {}"]
        param_string = "\n".join(param_parts) if linebreak else " ".join(param_parts)
        return param_string.format(self.alpha, self.gamma, self.epsilon_decay,
                                   self.epsilon_min, model_shape, self.tau)

    def decay_epsilon(self):
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def random_action(self):
        return self.env.action_space.sample()

    def training(self):
        return len(self.memory.memory) > self.warmup and self.step % self.train_interval == 0

    def best_action(self, state):
        q_vals = self.model.predict(state)[0]
        return np.argmax(q_vals)

    def explore(self):
        return np.random.random() < self.epsilon

    def select_action(self, state, test=False):
        action = self.random_action() if self.explore() and not test else self.best_action(state)
        self.decay_epsilon()
        self.step += 1
        return action

    def get_soft_updates(self):
        tw = self.target_model.get_weights()
        sw = self.model.get_weights()

        for i in range(len(tw)):
            tw[i] = sw[i] * self.tau + tw[i] * (1 - self.tau)
        self.target_model.set_weights(tw)

    def update_target(self):
        self.get_soft_updates()

    def save_model(self, fn):
        self.model.save(fn)

    def record(self, obs):
        self.memory.record(obs)

    def reset(self):
        self.step = 0

    def replay_memory(self):
        if not self.training():
            return

        # Loop through batch of experiences and update model weights
        experiences = self.memory.sample(self.batch_size)
        for experience in experiences:
            ss, aa, rr, ns, done = experience
            target = self.target_model.predict(ss)
            if done:
                target[0][aa] = rr
            else:
                target_q_value = max(self.target_model.predict(ns)[0])
                target[0][aa] = rr + target_q_value * self.gamma
            self.model.fit(ss, target, epochs=1, verbose=0)


class Experiment:
    def __init__(self, params, n_train=1000, n_test=100, gym_env="LunarLander-v2", render=False, id=None):

        self.episodes = n_train
        self.test_period = n_test
        self.reward_history = []
        self.epsilon_history = []
        self.max_reward = -float('inf')
        self.experiment_id = uuid.uuid1() if id is None else id

        self.model_architecture = params.get('model_architecture')
        self.alpha = params.get('alpha')
        self.gamma = params.get('gamma')
        self.ep_decay = params.get('ep_decay')
        self.ep_min = params.get('ep_min')
        self.batch_size = params.get('batch_size')
        self.warmup = params.get('warmup')
        self.tau = params.get('tau')
        self.memory_size = params.get('memory_size')

        self.render = render
        self.trial = 0
        self.trial_reward = 0
        self.max_trial_reward = -float('inf')

        # Create Environment
        self.env = gym.make(gym_env)
        self.shape = self.env.observation_space.shape[0]

        # Initialize Learner
        self.agent = DQNAgent(self.env, self.model_architecture, self.alpha, self.gamma, self.ep_decay,
                              self.ep_min, self.batch_size, self.warmup, self.tau, self.memory_size)

    def record_trial_stats(self):
        self.reward_history.append(self.trial_reward)
        self.epsilon_history.append(self.agent.epsilon)

        if self.trial_reward > self.max_trial_reward:
            self.max_trial_reward = self.trial_reward
            logging.info("New maximum reward! {} Trial: {}".format(self.trial_reward, self.trial))

            # Store Best
            fn = './models/best/experiment_{}_best.model'.format(self.experiment_id)
            self.agent.save_model(fn)

        if self.trial % 50 == 0:
            rh = pd.Series(self.reward_history, name="Mean Reward (window=100)")
            rhm = rh.rolling(window=100, min_periods=0).mean()
            ep = pd.Series(self.epsilon_history, name="Epsilon")

            agent_params = self.agent.get_param_string(linebreak=True)

            plt.figure(figsize=(14, 7))
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
            ind = range(len(self.reward_history))

            ax1.scatter(ind, self.reward_history, color='blue')
            ax1.plot(ind, rhm, 'g--')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward - 100 Episode MA')

            ax2 = ax1.twinx()
            ax2.plot(ind, ep, 'r:')
            ax2.set_ylim((0., 1.))
            ax2.set_ylabel('Epsilon')

            title_string = "Training - Trial {}".format(self.trial)
            plt.title(title_string)
            plt.figtext(0.99, 0.01, agent_params, horizontalalignment='right')

            plt.savefig('./plots/{}_training.png'.format(self.experiment_id))

        self.reset_stats()

    def record_experiment_stats(self):
        fn = './models/last/experiment_{}_last.model'.format(self.experiment_id)
        self.agent.save_model(fn)

    def record_episode_stats(self, reward):
        self.trial_reward += reward

    def reset_stats(self):
        self.trial_reward = 0

    def run_training_trial(self):
        self.agent.reset()
        ss = self.env.reset().reshape(1, self.shape)

        # Continue loop until done flag is triggered by environment
        while True:
            if self.render:
                self.env.render()

            # Take an action in the environment
            aa = self.agent.select_action(ss)
            ns, rr, done, _ = self.env.step(aa)
            ns = ns.reshape(1, self.shape)

            # Send to replay memory and replay if appropriate
            self.agent.record((ss, aa, rr, ns, done))
            self.agent.replay_memory()

            # Update the target model
            self.agent.update_target()

            self.record_episode_stats(rr)
            ss = ns

            if done:
                self.trial += 1
                if self.trial % 100 == 0:
                    print("{} HEARTBEAT - TRIAL {}".format(self.experiment_id, self.trial))
                break

    def run_testing_trials(self, fn=None):
        if fn is None:
            fn = './models/best/experiment_{}_best.model'.format(self.experiment_id)

        self.agent.reset()
        self.agent.set_model(fn)

        reward_history = []

        for trial in range(self.test_period):
            ss = self.env.reset().reshape(1, self.shape)
            total_reward = 0

            # Continue loop until done flag is triggered by environment
            while True:
                if self.render:
                    self.env.render()

                # Take an action in the environment
                aa = self.agent.select_action(ss, test=True)
                ns, rr, done, _ = self.env.step(aa)
                ns = ns.reshape(1, self.shape)
                total_reward += rr
                ss = ns

                if done:
                    break
            reward_history.append(total_reward)

        return reward_history

    def gen_train_results(self):
        rh = pd.Series(self.reward_history)
        df = pd.DataFrame(rh, index=range(self.episodes))
        df['experiment_id'] = self.experiment_id
        df.columns = ['reward', 'experiment_id']
        df.to_csv('./data/{}_training_results'.format(self.experiment_id))
        return df

    def gen_test_results(self, test_results):
        rh = pd.Series(test_results)
        df = pd.DataFrame(rh, index=range(self.test_period))
        df['experiment_id'] = self.experiment_id
        df.columns = ['reward', 'experiment_id']
        df.to_csv('./data/{}_test_results'.format(self.experiment_id))

        # Generate plots
        agent_params = self.agent.get_param_string(linebreak=True)
        plt.figure(figsize=(14, 7))
        plt.plot(rh)
        plt.scatter(rh.index, test_results)
        title_string = "Test Results | Mean Score: {}".format(rh.mean())
        plt.title(title_string)
        plt.figtext(0.99, 0.01, agent_params, horizontalalignment='right')
        plt.savefig('./plots/{}_test_results.png'.format(self.experiment_id))

        return df

    def train(self):
        for trial in range(self.episodes):
            self.run_training_trial()
            self.record_trial_stats()

        self.record_experiment_stats()
        return self.gen_train_results()

    def test(self, model_path=None):
        print("{} TESTING MODEL".format(self.experiment_id))
        test_results = self.run_testing_trials(model_path)
        self.gen_test_results(test_results)
        return


alphas = [0.001, 0.0001]
gammas = [0.95, 0.99]
taus = [0.1, 0.12]

n_trials = 1000
n_test = 100

all_params = []

for alpha in alphas:
    for gamma in gammas:
        for tau in taus:
            all_params.append({
                "model_architecture": [(48, 'relu'), (64, 'relu'), (48, 'relu')],
                "alpha": alpha,
                "tau": tau,
                "gamma": gamma,
                "ep_decay": 0.99995,
                "ep_min": 0.01,
                "batch_size": 24,
                "warmup": 24,
                "memory_size": 200000})


def execute_experiment(params):
    model_shape = "-".join([str(size) for size, _ in params['model_architecture']])
    experiment_name = "a_{}_g_{}_t_{}_{}".format(params['alpha'], params['gamma'], params['tau'], model_shape)
    ex = Experiment(params, n_train=n_trials, n_test=n_test, id=experiment_name)
    ex_rewards = ex.train()
    ex.test()
    return ex_rewards


def run():
    results = Parallel(n_jobs=10, verbose=0)(map(delayed(execute_experiment), all_params))
    all_res = pd.concat(results)
    all_res.to_csv('./data/experiment_results_{}'.format(uuid.uuid1()), index=False)


