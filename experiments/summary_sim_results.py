import pickle
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # low frequency
    rmses = list()
    for i in tqdm(range(10)):
        # file_name = "results/sim_freq_low_rank2_t{}.pkl".format(i)
        file_name = "results/full/sim_freq_low_rank2_t{}.pkl".format(i)
        with open(file_name, "rb") as f:
            res = pickle.load(f)
        rmses.append(res['rmse'])
    rmses = np.asarray(rmses)
    print("rmse = {}({})".format(np.mean(rmses), np.std(rmses)))

    # high frequency
    rmses = list()
    for i in range(10):
        # file_name = "results/sim_freq_high_rank2_t{}.pkl".format(i)
        file_name = "results/full/sim_freq_high_rank2_t{}.pkl".format(i)
        with open(file_name, "rb") as f:
            res = pickle.load(f)
        rmses.append(res['rmse'])
    rmses = np.asarray(rmses)
    print("rmse = {}({})".format(np.mean(rmses), np.std(rmses)))

    # varying frequency
    rmses = list()
    for i in range(10):
        # file_name = "results/sim_freq_varying_rank2_t{}.pkl".format(i)
        file_name = "results/full/sim_freq_varying_rank2_t{}.pkl".format(i)
        with open(file_name, "rb") as f:
            res = pickle.load(f)
        rmses.append(res['rmse'])
    rmses = np.asarray(rmses)
    print("rmse = {}({})".format(np.mean(rmses), np.std(rmses)))
