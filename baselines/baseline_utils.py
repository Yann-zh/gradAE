import random
import torch
import numpy as np


def set_seed(seed):
    print(f"setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0, 0.02)
        torch.nn.init.normal_(m.bias, 0, 0.02)


def dataLoading(path):
    filepath = path
    data = np.load(filepath)

    x = data['X']
    from sklearn.preprocessing import StandardScaler
    enc = StandardScaler()
    x = enc.fit_transform(x)
    print(x.shape)
    y = data['y']
    return x, y


def dataLoading_sample(path, n_samples_up_threshold):
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
        outlier_rate = sum(y) / X.shape[0]

        X_y0 = X[y == 0]
        X_y1 = X[y == 1]
        
        from sklearn.utils import resample
        X_y0_sampled = resample(X_y0, replace=True,
                                n_samples=int((1 - outlier_rate) * n_samples_up_threshold),
                                random_state=42)
        X_y1_sampled = resample(X_y1, replace=True, n_samples=int(outlier_rate * n_samples_up_threshold),
                                random_state=42)
        X = np.vstack((X_y0_sampled, X_y1_sampled))
        y = np.hstack((np.zeros(len(X_y0_sampled)), np.ones(len(X_y1_sampled))))
    return X, y
