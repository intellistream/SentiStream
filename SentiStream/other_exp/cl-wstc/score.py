import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

time_start=1
time_end=5
dataname = 'agnews'#'HuffN8'#

def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]
    
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)
          
    f = np.mean(fs).round(2)

    result = {}
    result['F1-score'] = f
    return result

total_pred=[]
total_truth=[]

eval_score=[]
acc=[]
len_every=[]

for time in range(time_start,time_end+1):
    dataset_dir='datasets/'+dataname+'_'+str(time)+'/'

    if time != time_end:
        truth_file = open(os.path.join(dataset_dir, 'accept_GT.txt'))
        truth= [i[:].strip('\n') for i in truth_file.readlines()]
        pred_file = open(os.path.join(dataset_dir, 'accept_pred.txt'))
        pred= [i[:].strip('\n') for i in pred_file.readlines()]
    else:
        truth_file = open(os.path.join(dataset_dir, 'test_labels.txt'))
        truth= [i[:].strip('\n') for i in truth_file.readlines()]
        pred_file = open(os.path.join(dataset_dir, 'out.txt')) 
        pred= [i[:].strip('\n') for i in pred_file.readlines()]

    cm_every = confusion_matrix(truth, pred)
    eval_score_every = F_measure(cm_every)['F1-score']
    acc_every = round(accuracy_score(truth, pred) * 100, 2)
    eval_score.append(eval_score_every)
    acc.append(acc_every)
    len_every.append(len(pred))

    total_pred= total_pred+pred
    total_truth= total_truth+truth

# macro-f1
print(len(total_pred))
cm_total = confusion_matrix(total_truth, total_pred)
eval_score_total = F_measure(cm_total)['F1-score']
acc_total = round(accuracy_score(total_truth, total_pred) * 100, 2)
print('f1_total:'+str(eval_score_total))
print('acc_total:'+str(acc_total))

len_aver=(len_every[0]*1+len_every[1]*2+len_every[2]*3+len_every[3]*4+len_every[4]*5)/len(total_pred)

print(len_every)
print(len_aver)
print(eval_score)
print(acc)
