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


# A plotter class to visualize the experiment results.
class Plotter:
    def __init__(self, **kwargs):
        self.data_dict = {}
        self.cur_dataset = ''
        self.colors = ['#ABC6E4', '#FCDABA', '#A7D2BA', '#D0CADE', '#C39398', '#F98F34', '#C2B1D7', '#1963B3']
        self.hps = {}
        self.file_path = f'ae/plots/'
        for key, value in kwargs.items():
            self.hps[key] = value
            self.file_path += f'{value}_'

    def __getitem__(self, key):
        return self.data_dict[self.cur_dataset][key]

    def __setitem__(self, key, value):
        self.data_dict[self.cur_dataset][key] = value

    def __contains__(self, key):
        return key in self.data_dict[self.cur_dataset][key]

    def create_entry(self, key, dataset):
        if dataset not in self.data_dict.keys():
            self.data_dict[dataset] = {}
        if key not in self.data_dict[dataset].keys():
            self.data_dict[dataset][key] = []

    def data_append(self, key, value, dataset=None):
        if dataset is None:
            dataset = self.cur_dataset
        self.create_entry(dataset=dataset, key=key)
        self.data_dict[dataset][key].append(value)

    def data_assign(self, key, value, dataset=None):
        if dataset is None:
            dataset = self.cur_dataset
        self.create_entry(dataset=dataset, key=key)
        self.data_dict[dataset][key] = value

    def set_cur_dataset(self, dataset):
        self.cur_dataset = dataset

    def save_all_data(self):
        for dataset in self.data_dict.keys():
            stat_dict = self.data_dict[dataset]
            num_epoch = len(stat_dict['AUC'])
            keys = list(stat_dict.keys())
            values = list(stat_dict.values())
            values = [value if len(value) == num_epoch else value * num_epoch for value in values]
            array_values = np.array(values)

            filename = os.path.join(self.file_path, f'{dataset}_all_stats.csv')
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Keys'] + keys)
                # print(f'Writing statistics of {dataset}. Keys:{keys}')
                for i in range(len(array_values[0])):
                    writer.writerow([i] + list(array_values[:, i]))

    def plot_early_stoppings(self):
        indexes = ['AUC', 'entropy',
                   'minus_inter', 'theta', 'o_num_ground_truth', 'i_num_ground_truth']
        fig, axs = plt.subplots(len(indexes), 1, sharex=True, figsize=(6, 0.8*len(indexes)))
        data = {key: self[key] for key in indexes}

        for ax, (key, values) in zip(axs, data.items()):
            x = np.arange(len(values))
            ax.plot(x, values, label=key)
            ax.set_ylabel(key)

        axs[0].axvline(self['EntropyStop'], color='#1F77B4', linestyle='--')
        axs[0].axvline(self['AUCStop'], color='blue', linestyle='--')
        axs[0].axvline(self['GradStop'], color='black', linestyle='--')
        axs[0].legend(loc="upper right")
        fig.tight_layout()
        make_plot_dir(f'{self.file_path}/early_stoppings/')
        plt.savefig(f'{self.file_path}/early_stoppings/{self.cur_dataset}-es.svg', format='svg')
        plt.close()

    def plot_whole_early_stoppings(self):
        datasets = list(self.data_dict.keys())
        metrics = {'Last': [], 'Mean': [], 'Max': [], 'AUCStop': [],
                   'EntropyStop': [], 'GradStop': []}
        born_effective = []
        zero_stop = []

        for dataset in datasets:
            self.set_cur_dataset(dataset)
            AUCs = self['AUC']
            metrics['Last'].append(AUCs[-1])
            metrics['Mean'].append(round(sum(AUCs) / len(AUCs), 3))
            metrics['Max'].append(max(AUCs))
            metrics['EntropyStop'].append(AUCs[self['EntropyStop'][0]])
            metrics['AUCStop'].append(AUCs[self['AUCStop'][0]])
            metrics['GradStop'].append(AUCs[self['GradStop'][0]])
            zero_stop.append(self['ZeroStop'][0])

        datasets.append('Mean')
        for key in metrics.keys():
            metrics[key].append(round(sum(metrics[key]) / len(metrics[key]), 3))

        fig, ax = plt.subplots(layout='constrained', figsize=(2 + 2 * len(datasets), 6))
        bar_width = 0.10
        i = np.arange(len(datasets))
        j = 0

        for metric, values in metrics.items():
            if metric == 'GradStop':
                color = ['blue' if z else self.colors[j] for z in zero_stop]
            else:
                color = self.colors[j]
            rects = ax.bar(i + j * bar_width, values, bar_width, label=metric, color=color)
            ax.bar_label(rects, padding=3)
            j += 1
        ax.set_ylabel('AUC value')
        ax.set_title('Experiments on Early Stopping Performance')
        ax.set_xticks(i + bar_width, datasets)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        make_plot_dir(f'{self.file_path}/whole_early_stoppings')

        plt.savefig(f'{self.file_path}/whole_early_stoppings/whole_es_performance.svg', format='svg')
        plt.close()
