import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

from ae.EntropyEarlyStop import getEntropyStopOffline
from ae.GradStop import getGradStopOffline
from ae.model.utils_ae import dataLoading, weights_init_normal, set_seed, get_ae_hps, cal_entropy
from ae.model.VanillaAE import Autoencoder
from plotter import Plotter, compute_theta

# Fixed parameters
k_list = [20]
sample_ratio_list = [10]
resample_list = [True]
resample_freq_list = [10]
seeds = [0, 1, 2]

# adjustable parameters, according to specific Algorithm-dataset pair on which GradStop is applied
theta_threshold_list = [1.57]
window_size = 20
R = 0.001
beneficial_threshold = 0.05
significance_threshold = 0.01


def naive_train(x, y, lr=0.001, epoch=100, plotter=None):
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

    for e in tqdm(range(epoch)):
        model.train()
        inlier_losses = []
        outlier_losses = []
        outlier_grads_epoch = []
        inlier_grads_epoch = []
        entropy = []
        inlier_inter = 0
        outlier_inter = 0

        batch_size = X.size(0)
        for i in range(0, X.size(0), batch_size):
            batch_x = train_X[i:i + batch_size]
            recon_score = model(batch_x)

            losses = recon_score
            loss = torch.mean(losses)
            numpy_recon_score = recon_score.cpu().detach().numpy()
            entropy = cal_entropy(numpy_recon_score)

            opt.zero_grad()

            inlier_grads_epoch = []
            outlier_grads_epoch = []
            all_grads = []
            all_labels = []
            for j in range(len(X)):
                if len(all_grads) >= sample_ratio * (k_i + k_o):
                    if len(outlier_grads_epoch) == 0:
                        while train_y[j] != 1:
                            j += 1
                    else:
                        break
                grad = torch.autograd.grad(recon_score[j], model.parameters(), retain_graph=True)
                grad_flatten = np.array([])
                for g in grad:
                    grad_flatten = np.append(grad_flatten, g.flatten().clone().detach().cpu().numpy(), axis=0)

                if train_y[j] == 1:
                    outlier_losses.append(recon_score[j].item())
                    outlier_grads_epoch.append(grad_flatten)
                else:
                    inlier_losses.append(recon_score[j].item())
                    inlier_grads_epoch.append(grad_flatten)
                all_grads.append(grad_flatten)
                all_labels.append(train_y[j])

            loss.backward()

            all_grads = np.array(all_grads)
            all_labels = np.array(all_labels)
            all_grads_norm = np.linalg.norm(all_grads, axis=1)

            if (resample and e % resample_freq == 0) or e == 0:
                inlier_indices = np.argsort(all_grads_norm)[:k_i]
                outlier_indices = np.argsort(-all_grads_norm)[:k_o]

            inlier_grads_epoch = all_grads[inlier_indices]
            outlier_grads_epoch = all_grads[outlier_indices]

            inlier_num_ground_truth = sum(all_labels[inlier_indices])
            outlier_num_ground_truth = sum(all_labels[outlier_indices])
            if e == 0:
                print(
                    f'Sampling done. #Inliers | #Outliers | #Total: {len(inlier_losses)} | {len(outlier_losses)} | {len(inlier_losses) + len(outlier_losses)}. '
                    f'Get last-{k_i}and top-{k_o} as sets G_last and G_top.')

            inlier_inter = np.linalg.norm(np.sum(inlier_grads_epoch, axis=0), axis=0) / np.sum(
                np.linalg.norm(inlier_grads_epoch, axis=1), axis=0)
            outlier_inter = np.linalg.norm(np.sum(outlier_grads_epoch, axis=0), axis=0) / np.sum(
                np.linalg.norm(outlier_grads_epoch, axis=1), axis=0)

            opt.step()

        # Compute and log statistics to plotter

        o_grad_sum = np.sum(outlier_grads_epoch, axis=0)

        i_grad_sum = np.sum(inlier_grads_epoch, axis=0)

        theta = compute_theta(o_grad_sum, i_grad_sum)

        plotter.data_append('o_inter', outlier_inter)
        plotter.data_append('o_num_ground_truth', outlier_num_ground_truth)

        plotter.data_append('entropy', entropy)

        if e != 0:
            num_avg = min(len(plotter['theta']), 8)
            plotter.data_append('theta', (sum(plotter['theta'][-num_avg:]) + theta) / (num_avg + 1))
            plotter.data_append('minus_inter',
                                (sum(plotter['minus_inter'][-num_avg:]) + inlier_inter - outlier_inter) / (num_avg + 1))
        else:
            plotter.data_append('theta', theta)
            plotter.data_append('minus_inter', inlier_inter - outlier_inter)

        plotter.data_append('i_inter', inlier_inter)
        plotter.data_append('i_num_ground_truth', inlier_num_ground_truth)

        with torch.no_grad():
            model.eval()
            numpy_recon_score = model(X).cpu().detach().numpy()
            auc = roc_auc_score(y, numpy_recon_score)
            auc_sample = roc_auc_score([0] * len(inlier_losses) + [1] * len(outlier_losses),
                                       inlier_losses + outlier_losses)
            plotter.data_append('AUC', round(auc, 3))
            plotter.data_append('AUC_sample', round(auc_sample, 3))

    if np.mean(plotter['theta'][5:10]) > theta_threshold:
        plotter.data_assign('GradStop', [0])
        plotter.data_assign('ZeroStop', [True])
    else:
        plotter.data_assign('GradStop', [getGradStopOffline(plotter['minus_inter'], k=window_size, r=R,
                                                            beneficial_threshold=beneficial_threshold,
                                                            significance_threshold=significance_threshold)])
        plotter.data_assign('ZeroStop', [False])
    plotter.data_assign('EntropyStop', [getEntropyStopOffline(plotter['entropy'], k=100, R_down=0.1)])
    plotter.data_assign('AUCStop',
                        [getEntropyStopOffline([1 - auc for auc in plotter['AUC_sample']], k=21, R_down=0.1)])

    plotter.plot_early_stoppings()
    plotter.plot_whole_early_stoppings()
    return auc, numpy_recon_score


if __name__ == '__main__':
    print(f"Model name:AE")
    template_model_name = 'ae'

    act, dropout, h_dim, num_layer, lr, epoch = get_ae_hps()  # get all the hp configs

    n_samples_up_threshold = 10000
    dry_run = False

    for seed in seeds:
        for k_o in k_list:
            k_i = k_o
            for sample_ratio in sample_ratio_list:
                for theta_threshold in theta_threshold_list:
                    for resample in resample_list:
                        for resample_freq in resample_freq_list:
                            print('k_o', 'k_i', 'sample_ratio',
                                  'theta_threshold', 'resample', 'resample_freq')
                            print(k_o, k_i, sample_ratio, theta_threshold,
                                  resample, resample_freq)
                            if os.path.exists(
                                    f'plots/{k_o}_{k_i}_{sample_ratio}_{theta_threshold}_{resample}_{resample_freq}_{seed}_'):
                                continue
                            plotter = Plotter(k_o=k_o, k_i=k_i,
                                              sample_ratio=sample_ratio,
                                              theta_threshold=theta_threshold, resample=resample,
                                              resample_freq=resample_freq, seed=seed)
                            g = os.walk(r"data/data_table")
                            for path, dir_list, file_list in g:
                                for file_name in file_list:
                                    try:
                                        if dry_run:
                                            epoch = 10
                                            if file_name not in [
                                                '1_ALOI.npz',
                                            ]:
                                                continue
                                        print(f'Training on：{file_name}')
                                        plotter.set_cur_dataset(file_name.split('.')[0])
                                        AUC = []
                                        Time = []

                                        data_path = os.path.join(path, file_name)
                                        x, y = dataLoading(data_path, n_samples_up_threshold=n_samples_up_threshold)

                                        print(file_name)
                                        print(f'hp: {act}-{dropout}-{h_dim}-{num_layer}-{lr}-{epoch}')

                                        set_seed(seed)
                                        auc, outlier_score = naive_train(x, y, lr, epoch, plotter=plotter)
                                        AUC.append(auc)
                                    except Exception as e:
                                        print(f'Dataset {file_name} failed.')
                                        print(f'{e.args}')
                                        import traceback

                                        traceback.print_exc()
                                        continue

                            plotter.save_all_data()
