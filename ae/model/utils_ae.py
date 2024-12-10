import torch
import numpy as np

import random


# For Entropy Stop from work EntropyStop: Unsupervised Deep Outlier Detection with Loss Entropy, KDD '24
def cal_entropy(score):
    score = score.reshape(-1)
    score = score / np.sum(score)  # to possibility
    entropy = np.sum(-np.log(abs(score) + 10e-8) * score)
    return entropy


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def dataLoading(path, n_samples_up_threshold):
    filepath = path
    data = np.load(filepath)

    X = data['X']
    # standardization
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    X = enc.fit_transform(X)
    print(X.shape)
    y = data['y']

    if X.shape[0] > n_samples_up_threshold:
        outlier_ratio = sum(y) / X.shape[0]
        print(f'Proportional sampling with outlier ratio：{round(outlier_ratio, 3)}，'
              f'#Inliers：{int((1 - outlier_ratio) * n_samples_up_threshold)}，'
              f'#Outliers：{int(outlier_ratio * n_samples_up_threshold)}。')
        X_y0 = X[y == 0]
        X_y1 = X[y == 1]

        from sklearn.utils import resample
        X_y0_sampled = resample(X_y0, replace=True,
                                n_samples=int((1 - outlier_ratio) * n_samples_up_threshold),
                                random_state=42)
        X_y1_sampled = resample(X_y1, replace=True, n_samples=int(outlier_ratio * n_samples_up_threshold),
                                random_state=42)
        X = np.vstack((X_y0_sampled, X_y1_sampled))
        y = np.hstack((np.zeros(len(X_y0_sampled)), np.ones(len(X_y1_sampled))))
    return X, y


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.normal_(m.bias.data, 0.0, 0.02)


def get_ae_hps():

    act_func = 'relu'
    dropout = 0.2
    h_dim = 64
    num_layer = 1
    lr = 0.005
    epochs = 100

    return act_func, dropout, h_dim, num_layer, lr, epochs
