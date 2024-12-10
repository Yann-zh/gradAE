import numpy as np
import copy


class GradStop:
    def __init__(self, k=21, r=0.1, beneficial_threshold=0.05, significance_threshold=0):
        self.k = k
        self.r = r
        self.grads = []
        self.i_inter = []
        self.o_inter = []
        self.patience = 0
        self.min_grad = None
        self.max_grad = None
        self.G = 0
        self.model = None
        self.max_grad_epoch = 0
        self.beneficial_threshold = beneficial_threshold
        self.significance_threshold = significance_threshold

    def step(self, grad):
        self.grads.append(grad)
        if self.min_grad is None:
            self.min_grad = grad
            self.max_grad = grad
            return False

        self.G += abs(grad - self.grads[-2])

        if (
                (grad > self.max_grad and (grad - self.max_grad) / self.G > self.r)
                or
                grad > self.beneficial_threshold
                or
                abs(grad) < self.significance_threshold
        ):

            self.max_grad = grad
            self.max_grad_epoch = len(self.grads) - 1
            self.patience = 0
            self.G = 0
        else:
            self.patience += 1

        if self.patience >= self.k:
            return True
        return False

    def getBestEpoch(self):
        return self.max_grad_epoch


def getGradStopOffline(grads, k=11, r=0.01, beneficial_threshold=0.05, significance_threshold=0):  # offline version
    GS = GradStop(k, r, beneficial_threshold, significance_threshold)
    for i in range(len(grads)):
        isStop = GS.step(grads[i])
        if isStop:
            break
    return GS.getBestEpoch()
