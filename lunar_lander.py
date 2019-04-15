from agent_utils import Experiment
from joblib import Parallel, delayed
import pandas as pd
import uuid

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


if __name__ == "__main__":
    run()
