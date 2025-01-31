import torch
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
from tqdm import tqdm

from ae.model.utils_ae import dataLoading
from Ensemble.randnet import LinearRandNet

def run_one(path,file_name):
    # data_path = "/home/mist/od-data/20_letter.npz"
    data_path = os.path.join(path, file_name)
    x, y = dataLoading(data_path, n_samples_up_threshold=10000)
    

    model = LinearRandNet(input_dim_list=[x.shape[-1],64],learning_rate=0.005,epochs=100,pre_epochs=0,device='cuda',batch_size=x.shape[0])
    X = torch.FloatTensor(x)
    rt,mem,_ = model.fit(X)
    score =model.predict(X)

    auc = roc_auc_score(y,score)
    ap = average_precision_score(y,score)
    print('auc:',auc)
    return auc,ap,rt,mem


if __name__ == '__main__':
    template_model_name = 'RandNet_full'
    try_times = 3
    for th in range(try_times):
        g = os.walk(r"./data")
        cnt =0
        Auc = []
        Dataset = []
        Ap =[]
        Mem = []
        RunTime = []

        for path,dir_list,file_list in g:
            for file_name in file_list:
                print(file_name)
                print(cnt)
                cnt +=1
                auc,ap,rt,mem = run_one(path,file_name)

                Auc.append(auc)
                Dataset.append(file_name)
                Ap.append(ap)
                RunTime.append(rt)
                Mem.append(mem)



        df = pd.DataFrame({'Dataset':Dataset,
                           "AUC":Auc,
                           "AP":Ap,
                           "RunTime":RunTime,
                           "Memory":Mem
                           })
        df.to_csv(f'./res/{template_model_name}-{th}.csv', index=False)

    import subprocess
    subprocess.run('~/code/send_msg.sh "randnet done"',shell=True,capture_output=False)
