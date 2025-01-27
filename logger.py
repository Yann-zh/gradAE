from matplotlib import pyplot as plt
from numpy.linalg import norm
import numpy as np
import csv
import os


# from warnings import simplefilter
#
# simplefilter('error')


def compute_theta(a, b):
    # print('此时的cosv是：',np.dot(a, b) / (norm(a) * norm(b)))
    cos_value = np.dot(a, b) / (norm(a) * norm(b))
    theta_value = np.arccos(cos_value)
    return theta_value


def make_plot_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class Logger:
    def __init__(self, **kwargs):
        self.data_dict = {}
        self._cur_dataset = ''
        self.colors = ['#ABC6E4', '#FCDABA', '#A7D2BA', '#D0CADE', '#C39398', '#F98F34', '#C2B1D7', '#1963B3']
        self.hps = {}
        self.file_path = f'ae/results/'
        for key, value in kwargs.items():
            self.hps[key] = value
            self.file_path += f'{value}_'

    def __getitem__(self, key):
        return self.data_dict[self._cur_dataset][key]

    def __setitem__(self, key, value):
        self.data_dict[self._cur_dataset][key] = value

    def __contains__(self, key):
        return key in self.data_dict[self._cur_dataset][key]

    def create_entry(self, key, dataset):
        if dataset not in self.data_dict.keys():
            self.data_dict[dataset] = {}
        if key not in self.data_dict[dataset].keys():
            self.data_dict[dataset][key] = []

    def data_append(self, key, value, dataset=None):
        if dataset is None:
            dataset = self._cur_dataset
        self.create_entry(dataset=dataset, key=key)
        self.data_dict[dataset][key].append(value)

    def data_assign(self, key, value, dataset=None):
        if dataset is None:
            dataset = self._cur_dataset
        self.create_entry(dataset=dataset, key=key)
        self.data_dict[dataset][key] = value

    def set_cur_dataset(self, dataset):
        self._cur_dataset = dataset

    def get_cur_dataset(self):
        return self._cur_dataset

    def save_all_data(self):
        for dataset in self.data_dict.keys():
            stat_dict = self.data_dict[dataset]
            num_epoch = len(stat_dict['AUC'])
            keys = list(stat_dict.keys())
            values = list(stat_dict.values())
            values = [value if len(value) == num_epoch else value * num_epoch for value in values]
            array_values = np.array(values)

            filename = os.path.join(self.file_path, f'{dataset}_all_stats.csv')
            if not os.path.exists(self.file_path):
                os.makedirs(self.file_path)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Keys'] + keys)
                # print(f'Writing statistics of {dataset}. Keys:{keys}')
                for i in range(len(array_values[0])):
                    writer.writerow([i] + list(array_values[:, i]))
