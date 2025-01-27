import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from ae.EntropyEarlyStop import getEntropyStopOffline
from ae.GradStop_pretty import getGradStopOffline
from ae.model.utils_ae import dataLoading, weights_init_normal, set_seed, get_ae_hps, cal_entropy
from ae.model.VanillaAE import Autoencoder
from logger import Logger, compute_theta
import traceback

# Fixed parameters
k = 20
B_eval_size = 400
resample = True
resample_freq = 10
seeds = [0, 1, 2]

# adjustable parameters, according to specific Algorithm-dataset pair on which GradStop is applied
t_D = 1.57
window_size = 20
R_down = 0.001
t_C_B = 0.05
t_C_S = 0.01


def naive_train(x, y, lr=0.001, epoch=100, logger=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device：{device}')
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)

    model = Autoencoder(X.size(1)).to(device)
    model.apply(weights_init_normal)
    opt = Adam(model.parameters(), lr=lr)

    from tqdm import tqdm

    # shuffle x index
    random_index = np.arange(X.size(0))
    np.random.shuffle(random_index)
    train_X = X[random_index]
    train_y = y[random_index]

    inlier_num = len(y) - sum(y)
    outlier_num = sum(y)
    print(f'#Inliers：{inlier_num}, #Outliers：{outlier_num}')

    last_k_indices = []
    top_k_indices = []

    for e in tqdm(range(epoch)):
        model.train()

        G_top = []
        G_last = []

        C_last = 0
        C_top = 0

        batch_size = X.size(0)
        for i in range(0, X.size(0), batch_size):
            batch_x = train_X[i:i + batch_size]
            recon_score = model(batch_x)

            losses = recon_score
            loss = torch.mean(losses)

            opt.zero_grad()

            all_grads = []
            all_labels = []
            for j in range(len(X)):
                if len(all_grads) >= B_eval_size:
                    if sum(all_labels) == 0:
                        while train_y[j] != 1:
                            j += 1
                    else:
                        break
                grad = torch.autograd.grad(recon_score[j], model.parameters(), retain_graph=True)
                grad_flatten = np.array([])
                for g in grad:
                    grad_flatten = np.append(grad_flatten, g.flatten().clone().detach().cpu().numpy(), axis=0)

                all_grads.append(grad_flatten)
                all_labels.append(train_y[j])

            loss.backward()

            all_grads = np.array(all_grads)
            all_grads_norm = np.linalg.norm(all_grads, axis=1)

            if (resample and e % resample_freq == 0) or e == 0:
                last_k_indices = np.argsort(all_grads_norm)[:k_i]
                top_k_indices = np.argsort(-all_grads_norm)[:k_o]

            G_last = all_grads[last_k_indices]
            G_top = all_grads[top_k_indices]

            if e == 0:
                print(f'\nGradSample done. Get last-{k_i} and top-{k_o} as sets G_last and G_top.\n')

            C_last = np.linalg.norm(np.sum(G_last, axis=0), axis=0) / np.sum(
                np.linalg.norm(G_last, axis=1), axis=0)
            C_top = np.linalg.norm(np.sum(G_top, axis=0), axis=0) / np.sum(
                np.linalg.norm(G_top, axis=1), axis=0)

            opt.step()

        # Compute and log statistics to plotter

        o_grad_sum = np.sum(G_top, axis=0)

        i_grad_sum = np.sum(G_last, axis=0)

        D = compute_theta(o_grad_sum, i_grad_sum)

        logger.data_append('C_top', C_top)
        logger.data_append('C_last', C_last)

        if e != 0:
            num_avg = min(len(logger['D']), 8)  # Smoothing against fluctuation
            logger.data_append('D', (sum(logger['D'][-num_avg:]) + D) / (num_avg + 1))
            logger.data_append('C_diff',
                               (sum(logger['C_diff'][-num_avg:]) + C_last - C_top) / (num_avg + 1))
        else:
            logger.data_append('D', D)
            logger.data_append('C_diff', C_last - C_top)

        with torch.no_grad():
            model.eval()
            numpy_recon_score = model(X).cpu().detach().numpy()
            auc = roc_auc_score(y, numpy_recon_score)
            logger.data_append('AUC', round(auc, 3))

        logger.data_assign('GradStopEpoch', [getGradStopOffline(logger['C_diff'], logger['D'], window_size=window_size,
                                                           R_down=R_down, t_C_B=t_C_B, t_C_S=t_C_S, t_D=1.57)])

    return logger['AUC'][logger['GradStopEpoch'][0]], numpy_recon_score


if __name__ == '__main__':
    print(f"Model name:AE")
    template_model_name = 'ae'

    act, dropout, h_dim, num_layer, lr, epoch = get_ae_hps()  # get all the hp configs

    # Down sample datasets to 10000 for ones that larger than 10000
    n_samples_up_threshold = 10000
    dry_run = False

    all_AUCs = {}
    g = os.walk(r"data/data_table")
    for _, _, file_names in g:
        for file_name in file_names:
            all_AUCs[file_name.split('.')[0]] = []

    for seed in seeds:
        k_i = k_o = k
        print('k_o', 'k_i', 'B_eval_size', 't_D', 't_C_B', 't_C_S', 'resample', 'resample_freq')
        print(k_o, k_i, B_eval_size, t_D, t_C_B, t_C_S, resample, resample_freq)
        if os.path.exists(
                f'results/{t_D}_{t_C_B}_{t_C_S}_{seed}_'):
            continue
        logger = Logger(t_D=t_D, t_C_B=t_C_B, t_C_S=t_C_S, seed=seed)

        g = os.walk(r"data/data_table")
        for path, dirs, file_list in g:
            for file_name in file_list:
                try:
                    if dry_run:
                        epoch = 10
                        if file_name not in [
                            '1_ALOI.npz',
                        ]:
                            continue
                    print(f'Training on：{file_name}')
                    logger.set_cur_dataset(file_name.split('.')[0])

                    data_path = os.path.join(path, file_name)
                    x, y = dataLoading(data_path, n_samples_up_threshold=n_samples_up_threshold)

                    print(file_name)
                    print(f'hp: {act}-{dropout}-{h_dim}-{num_layer}-{lr}-{epoch}')

                    set_seed(seed)
                    auc, outlier_score = naive_train(x, y, lr, epoch, logger=logger)
                    all_AUCs[logger.get_cur_dataset()].append(auc)
                except Exception as e:
                    print(f'Failed. Seed: {seed}, Dataset: {file_name}.')
                    print(f'{e.args}')
                    traceback.print_exc()
                    continue

        logger.save_all_data()

    mean_AUCs = []
    for dataset, AUCs in all_AUCs.items():
        print(f'AUC on dataset {dataset}: {round(np.mean(AUCs), 3)}')
        mean_AUCs.append(np.mean(AUCs))

    print(f'Averaged AUC is {round(np.mean(mean_AUCs), 3)}')
