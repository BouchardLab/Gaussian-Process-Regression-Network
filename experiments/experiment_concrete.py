import os
import sys
import argparse
from hdf5storage import savemat
import numpy as np

sys.path.append('/mnt/d/Research/others/GPRN/code')
sys.path.append('/mnt/d/Data/mts_data')
# print(sys.path)

from GPRN import GPRN
from DataProcess import process


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

    signature = domain + '_rank' + str(rank) + '_t' + str(trial)
    cfg = {
        'data': data,
        'signature': signature,
        'jitter': 1e-8,
        'epochs': maxIter,
        'Q': rank,
        'kernel': kernel,
        'lr': 0.001,
        'record_time': True
    }

    model = GPRN(cfg)
    res = model.fit()
    # compute the MAE
    mae = np.mean(np.abs(res["Y_pred"][-1] - data["Y_test_ground"]))
    print(mae)
    # breakpoint()

    cfg['result'] = res
    res_save_path = os.path.join(res_path, signature)
    savemat(res_save_path, cfg, format='7.3')
    print('results saved to', res_save_path + '.mat')


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args = dict()
    args["domain"] = "concrete"
    args["kernel"] = "rbf"
    args["device"] = "0"
    args["rank"] = 2
    args["maxIter"] = 5000
    args["interval"] = 200

    for trial in range(5):
        args["trail"] = trial
        run(args)