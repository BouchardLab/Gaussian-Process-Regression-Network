import os
import sys
import argparse
from hdf5storage import savemat
import numpy as np
import pickle

sys.path.append('/mnt/d/Research/others/GPRN/code')
sys.path.append('/mnt/d/Data/mts_data')
# print(sys.path)

# catscan
# sys.path.append('/home/rmeng/Data/GPRN/code')
# sys.path.append('/home/rmeng/Data/Data/mts_data')

# uta
sys.path.append('/home/fan/Research/GPRN/code')
sys.path.append('/home/fan/Research/Data/mts_data')


from GPRN_miss import GPRN
from DataProcessMiss import process


def run(args):
    domain = args["domain"]
    kernel = args["kernel"]
    device = args["device"]
    rank = args["rank"]
    maxIter = args["maxIter"]
    interval = args["interval"]

    print('Experiment summary: ')
    print(' - Domain name:', domain)
    print(' - Device id:', device)
    print(' - Cov Func:', kernel)
    print(' - rank:', rank)
    print(' - maxIter:', maxIter)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    print('Using GPU', device)

    res_path = 'results'

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    trial = args["trail"]
    data = process(domain)

    # breakpoint()

    signature = domain + '_rank' + str(rank) + '_t' + str(trial)
    cfg = {
        'data': data,
        'signature': signature,
        'jitter': 1e-8,
        'epochs': maxIter,
        'Q': rank,
        'kernel': kernel,
        'lr': 0.1,
        'record_time': True
    }

    model = GPRN(cfg)
    res = model.fit()
    # compute the MAE
    # breakpoint()
    Y_test = [y_d * data["Y_std"] + data["Y_mean"] for y_d in data["Y_test"]]
    mae = [np.abs(y_test_p.reshape(-1) - y_pred_p.reshape(-1)) for y_test_p, y_pred_p in zip(Y_test, res["Y_pred"][-1])]
    mae = np.mean(np.concatenate(mae))
    se_list = [(y_test_p.reshape(-1) - y_pred_p.reshape(-1))**2 for y_test_p, y_pred_p in zip(Y_test, res["Y_pred"][-1])]
    rmse = np.sqrt(np.mean(np.concatenate(se_list)))
    print("mae", mae, "rmse", rmse)
    # breakpoint()

    cfg['result'] = res
    cfg['mae'] = mae
    cfg['rmse'] = rmse
    res_save_path = os.path.join(res_path, signature)
    # savemat(res_save_path, cfg, format='7.3')
    # print('results saved to', res_save_path + '.mat')
    results = dict()
    results['mae'] = mae
    results['rmse'] = rmse
    with open(res_save_path + ".pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args = dict()
    args["domain"] = "sim_freq_low"
    args["kernel"] = "rbf"
    args["device"] = "cuda"
    args["rank"] = 2
    args["maxIter"] = 2000
    args["interval"] = 200

    for trial in range(10):
        args["trail"] = trial
        run(args)