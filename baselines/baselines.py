from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.gmm import GMM
from pyod.models.cblof import CBLOF

from pyod.models.deep_svdd import DeepSVDD
from pyod.models.auto_encoder import AutoEncoder
from baseline_utils import dataLoading, dataLoading_sample, set_seed


import os
import numpy as np
import csv

from sklearn.metrics import roc_auc_score

n_samples_up_threshold = 10000
seeds = [0, 1, 2]

class PYOD_pipeline():
    def __init__(self, seed, model_name):
        self.seed = seed
        self.model_name = model_name

        self.model_dict = {'IForest': IForest, 'ECOD': ECOD, 'KNN': KNN, 'GMM': GMM, 'CBLOF': CBLOF,
                           'DeepSVDD': DeepSVDD, 'OCSVM': OCSVM, 'AutoEncoder': AutoEncoder}
        self.model = None

    def fit(self, X_train):
        model_class = eval(self.model_name)
        if self.model_name == 'IForest':
            self.model = model_class(n_estimators=100).fit(X_train)

        elif self.model_name == 'ECOD':
            self.model = model_class(contamination=0.1, n_jobs=1).fit(X_train)

        elif self.model_name == 'KNN':
            self.model = model_class().fit(X_train)

        elif self.model_name == 'OCSVM':
            self.model = model_class().fit(X_train)

        elif self.model_name == 'DeepSVDD':
            self.model = model_class(epochs=100).fit(X_train)

        elif self.model_name == 'GMM':
            self.model = model_class().fit(X_train)

        elif self.model_name == 'CBLOF':
            self.model = model_class().fit(X_train)

        else:
            raise NotImplementedError

    # from pyod: for consistency, outliers are assigned with larger anomaly scores
    def predict_score(self, X):
        score = self.model.decision_function(X)
        return score


def save_dict_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in data.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    UOD_baselines = ['IForest',
                     # 'ECOD', 'KNN', 'CBLOF', 'GMM', 'OCSVM'
                     ]
    performance_dict = {}
    for seed in seeds:
        try:
            for baseline in UOD_baselines:
                performance_dict = {}
                print(f'Runnning {baseline}.')
                g = os.walk(r"../data/data_table")
                for path, dir_list, file_list in g:
                    dataset_aucs = []
                    for file_name in file_list:
                        aucs = []
                        try:
                            print(f'Training on：{file_name}')
                            AUC = []
                            Time = []
                            data_path = os.path.join(path, file_name)
                            x, y = dataLoading_sample(data_path, n_samples_up_threshold=n_samples_up_threshold)
                            model = PYOD_pipeline(seed=seed, model_name=baseline)
                            model.fit(x)
                            score = model.predict_score(x)
                            auc = roc_auc_score(y, score)
                            aucs.append(auc)
                        except Exception as e:
                            print(f'Dataset {file_name} failed.')
                            print(f'{e.args}')
                            import traceback
                            traceback.print_exc()
                            continue
                        auc = np.mean(aucs)
                        dataset_number = file_name.split('_')[0]
                        performance_dict[f'{dataset_number}'] = auc
                        dataset_aucs.append(auc)
                    performance_dict[f'mean'] = np.mean(dataset_aucs)
                print(f'Saving to baseline_performance_{seed}.csv')
                save_dict_to_csv(performance_dict, f'{baseline}_performance_{seed}_unlimited.csv')
        except KeyboardInterrupt as e:
            print(f'Interrupted.')
            continue
