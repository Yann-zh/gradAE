import numpy as np
import copy


def moving_average_filter(data=None, window_size=21):
    assert window_size % 2 == 1
    if data is None:
        data = []
    padded_data = np.pad(data, (window_size - 1) // 2, mode='edge')
    weights = np.repeat(1.0, window_size) / window_size
    sma = np.convolve(padded_data, weights, 'valid')
    return list(data - sma)


class GradStop:
    '''
    This entropystop version will record the best iteration index when it encounters the minimum entropy. 
    If you need to get the outlier score, you need to store the outlier scores outside the function.
    '''

    def __init__(self, window_size=20, r=0.1, t_C_B=0.05, t_C_S=0, t_D=1.57):
        self.window_size = window_size
        self.R_down = r
        self.t_C_B = t_C_B
        self.t_C_S = t_C_S
        self.t_D = t_D

        self.C_diffs = []
        self.Ds = []

        self.C_last = []
        self.C_top = []
        self.patience = 0
        self.C_diff_max = None
        self.max_C_diff_epoch = 0
        self.H = 0
        self.model = None

    def step(self, C_diff, D):
        self.C_diffs.append(C_diff)
        self.Ds.append(D)
        if self.C_diff_max is None:
            self.C_diff_max = C_diff
            return False

        if (len(self.Ds) == self.window_size // 2 and
                np.mean(self.Ds[self.window_size // 4:self.window_size // 2]) > self.t_D):
            # Check if G_last and G_top are inherently divergent during the very early stage of training
            self.max_C_diff_epoch = 0
            return True

        self.H += abs(C_diff - self.C_diffs[-2])

        if (
                (C_diff > self.C_diff_max and (C_diff - self.C_diff_max) / self.H > self.R_down)
                or
                C_diff > self.t_C_B
                or
                abs(C_diff) < self.t_C_S
        ):

            self.C_diff_max = C_diff
            self.max_C_diff_epoch = len(self.C_diffs) - 1
            self.patience = 0
            self.H = 0
        else:
            self.patience += 1

        if self.patience >= self.window_size:
            return True
        return False

    def getBestEpoch(self):
        return self.max_C_diff_epoch


def getGradStopOffline(C_diffs, Ds, window_size=20, R_down=0.01, t_C_B=0.05, t_C_S=0, t_D=1.57):  # offline version
    GS = GradStop(window_size, R_down, t_C_B, t_C_S)
    for i in range(len(C_diffs)):
        isStop = GS.step(C_diffs[i], Ds[i])
        if isStop:
            break
    return GS.getBestEpoch()
