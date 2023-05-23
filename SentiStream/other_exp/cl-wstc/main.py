import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from threshold import *
from seed import *

from time import time as time_time

time_start = 1
time_end = 4
dataname = 'cmb'

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

    f = np.mean(fs).round(4)
    result = {}
    result['F1-score'] = f
    return result


start_time = time_time()

for time in range(time_start, time_end):

    print('peroid:'+str(time))

    dataset_dir = 'datasets/'+dataname+'_'+str(time)+'/'
    dataset_dir_before = 'datasets/'+dataname+'_'+str(time-1)+'/'
    dataset_dir_next = 'datasets/'+dataname+'_'+str(time+1)+'/'

    # LOTClass
    del_pt(dataset_dir)
    os.system('python src/train.py --dataset_dir '+dataset_dir)
    copy_txt(dataset_dir, 'category_vocab.txt', 'category_vocab_std.txt')
    copy_txt(dataset_dir, 'out.txt', 'out_std.txt')
    copy_txt(dataset_dir, 'confidence.txt', 'confidence_std.txt')

    if (time == time_end):
        break

    # threshold
    if (time > time_start):
        os.makedirs(dataset_dir+'threshold/')
        copy_txt_2(dataset_dir_before, 'threshold.txt',
                   dataset_dir+'threshold/', 'threshold_before.txt')
        copy_txt_2(dataset_dir, 'train.txt',
                   dataset_dir+'threshold/', 'train.txt')
        copy_txt_2(dataset_dir, 'label_names.txt',
                   dataset_dir+'threshold/', 'label_names.txt')
        copy_txt_2(dataset_dir_before, 'test.txt',
                   dataset_dir+'threshold/', 'test.txt')
        copy_txt_2(dataset_dir_before, 'test_labels.txt',
                   dataset_dir+'threshold/', 'test_labels.txt')
        copy_txt_2(dataset_dir_before, 'out.txt',
                   dataset_dir+'threshold/', 'out_before.txt')
        copy_txt_2(dataset_dir_before, 'confidence.txt',
                   dataset_dir+'threshold/', 'confidence_before.txt')
        threshold_update(dataset_dir, dataset_dir +
                         'threshold/', 0.7)  # ,dataname,time
    threshold_file = open(os.path.join(dataset_dir, 'threshold.txt'))
    threshold = [i[:].strip('\n') for i in threshold_file.readlines()]
    threshold_file.close()
    print('threshold', threshold)

    # accept reject
    confidence_file = open(os.path.join(dataset_dir, 'confidence_std.txt'))
    confidence = [i[:].strip('\n') for i in confidence_file.readlines()]
    confidence_file.close()
    test_file = open(os.path.join(dataset_dir, 'test.txt'), encoding='utf-8')
    test = [i[:].strip('\n') for i in test_file.readlines()]
    test_file.close()
    test_labels_file = open(os.path.join(dataset_dir, 'test_labels.txt'))
    test_labels = [i[:].strip('\n') for i in test_labels_file.readlines()]
    test_labels_file.close()
    out_file = open(os.path.join(dataset_dir, 'out_std.txt'))
    out = [i[:].strip('\n') for i in out_file.readlines()]
    out_file.close()
    ac_GT_out = open(os.path.join(dataset_dir, 'accept_GT.txt'), 'w')
    ac_pred_out = open(os.path.join(dataset_dir, 'accept_pred.txt'), 'w')
    re_out = open(os.path.join(dataset_dir, 'reject_text.txt'),
                  'w', encoding='utf-8')
    re_GT_out = open(os.path.join(dataset_dir, 'reject_GT.txt'), 'w')
    ac_truth = []
    ac_pred = []

    for i in range(len(confidence)):
        if (confidence[i] > threshold[int(out[i])]):
            ac_truth.append(test_labels[i])
            ac_GT_out.write(test_labels[i] + '\n')
            ac_pred.append(out[i])
            ac_pred_out.write(out[i] + '\n')
        else:
            re_out.write(test[i] + '\n')
            re_GT_out.write(test_labels[i] + '\n')
    ac_GT_out.close()
    ac_pred_out.close()
    re_out.close()
    re_GT_out.close()

    # macro-f1
    print('ACC LEN', len(ac_pred))
    cm_ac = confusion_matrix(ac_truth, ac_pred)
    eval_score_ac = F_measure(cm_ac)['F1-score']
    cm_total = confusion_matrix(test_labels, out)
    eval_score_total = F_measure(cm_total)['F1-score']
    acc_ac = round(accuracy_score(ac_truth, ac_pred) * 100, 2)
    acc_total = round(accuracy_score(test_labels, out) * 100, 2)
    print('f1_ac:'+str(eval_score_ac))
    print('f1_total:'+str(eval_score_total))
    print('acc_ac:'+str(acc_ac))
    print('acc_total:'+str(acc_total))

    # write
    f_out = open(os.path.join(dataset_dir, 'score.txt'), 'w')
    f_out.write('time:'+str(time)+'\n')
    f_out.write('f1_ac:'+str(eval_score_ac)+'\n')
    f_out.write('f1_total:'+str(eval_score_total)+'\n')
    f_out.write('acc_ac:'+str(acc_ac)+'\n')
    f_out.write('acc_total:'+str(acc_total)+'\n')
    print('TIME TAKEN', time_time() - start_time)
    f_out.close()

    # delay
    re_file = open(os.path.join(
        dataset_dir, 'reject_text.txt'), encoding='utf-8')
    re = [i[:].strip('\n') for i in re_file.readlines()]
    re_label_file = open(os.path.join(dataset_dir, 'reject_GT.txt'))
    re_label = [i[:].strip('\n') for i in re_label_file.readlines()]
    test_next_out = open(os.path.join(
        dataset_dir_next, 'test.txt'), 'a', encoding='utf-8')
    test_label_next_out = open(os.path.join(
        dataset_dir_next, 'test_labels.txt'), 'a')
    train_next_out = open(os.path.join(
        dataset_dir_next, 'train.txt'), 'a', encoding='utf-8')
    for text in re:
        test_next_out.write(''.join(text) + '\n')
        train_next_out.write(''.join(text) + '\n')
    for label in re_label:
        test_label_next_out.write(''.join(label) + '\n')
    re_file.close()
    re_label_file.close()
    test_next_out.close()
    train_next_out.close()  # !!!!!
    test_label_next_out.close()

    # category_vocab_accept.txt
    os.makedirs(dataset_dir+'seed/')
    copy_txt_2(dataset_dir, 'threshold.txt',
               dataset_dir+'seed/', 'threshold.txt')
    copy_txt_2(dataset_dir, 'train.txt', dataset_dir+'seed/',
               'train.txt')  # copy_txt_2(dataset_dir,'accecpt.txt'!!!!!
    copy_txt_2(dataset_dir, 'label_names.txt',
               dataset_dir+'seed/', 'label_names.txt')
    copy_txt_2(dataset_dir, 'test.txt', dataset_dir+'seed/', 'test.txt')
    copy_txt_2(dataset_dir, 'test_labels.txt',
               dataset_dir+'seed/', 'test_labels.txt')
    os.system('python src/train2.py --dataset_dir '+dataset_dir+'seed/')
    copy_txt_2(dataset_dir+'seed/', 'category_vocab.txt',
               dataset_dir, 'category_vocab_accept.txt')

    # seed  time<3,nodelete--flag=0 time>3--flag=1
    flag = 0 if (time-time_start+1 < 3) else 1
    # Select the seed word from the k+1 --> k=0
    seed_update(dataset_dir, 0, 6, flag)
    copy_txt_2(dataset_dir, 'label_names_next.txt',
               dataset_dir_next, 'label_names.txt')
