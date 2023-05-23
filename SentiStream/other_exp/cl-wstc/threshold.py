import math
import os

def df_accept_open(acc_re_before,out_before,confidence,topic_id,threshold):
    n1=0
    n2=0
    for i in range(len(acc_re_before)):      
        if(topic_id==int(out_before[i])):
            if(0==acc_re_before[i]):
                if(float(confidence[i])>1/(1+math.pow(math.e,-threshold))):
                    n1=n1+1
                else:
                    n2=n2+1
    return (-n1+n2)*1/(1+math.pow (math.e,-threshold))*(1-1/(1+math.pow(math.e,-threshold)))

def threshold_open(acc_re_before,out_before,confidence,threshold_before):
    iter_num=500
    eta=0.002
    threshold_new=[]
    for i in range(len(threshold_before)):
        eps=-math.log(1/float(threshold_before[i])-1)
        for j in range(iter_num):
            # print(1/(1+math.pow(math.e,-eps)))
            eps=eps-eta*df_accept_open(acc_re_before,out_before,confidence,i,eps)
        threshold_new.append(1/(1+math.pow(math.e,-eps)))
    return threshold_new

def df_accept_acc(acc_re_before,out_before,confidence,topic_id,threshold):
    n1=n2=n3=n4=0
    for i in range(len(acc_re_before)):      
        if(topic_id==int(out_before[i])):
            if(0==acc_re_before[i]):
                if(float(confidence[i])>1/(1+math.pow(math.e,-threshold))):
                    n1=n1+1
                else:
                    n2=n2+1
            else:
                if(float(confidence[i])>1/(1+math.pow(math.e,-threshold))):
                    n3=n3+1
                else:
                    n4=n4+1
    return (n2-n3)*1/(1+math.pow (math.e,-threshold))*(1-1/(1+math.pow(math.e,-threshold)))

def threshold_acc(acc_re_before,out_before,confidence,threshold_before):
    iter_num=500
    eta=0.002
    threshold_new=[]
    for i in range(len(threshold_before)):
        eps=-math.log(1/float(threshold_before[i])-1)
        for j in range(iter_num):
            # print(1/(1+math.pow(math.e,-eps)))
            eps=eps-eta*df_accept_acc(acc_re_before,out_before,confidence,i,eps)
        threshold_new.append(1/(1+math.pow(math.e,-eps)))
    return threshold_new

def df_combine_acc(acc_re_before,out_before,confidence,topic_id,threshold,alpha):
    n1=n2=n3=n4=0
    for i in range(len(acc_re_before)):      
        if(topic_id==int(out_before[i])):
            if(0==acc_re_before[i]):
                if(float(confidence[i])>1/(1+math.pow(math.e,-threshold))):
                    n1=n1+1
                else:
                    n2=n2+1
            else:
                if(float(confidence[i])>1/(1+math.pow(math.e,-threshold))):
                    n3=n3+1
                else:
                    n4=n4+1
    return (-alpha*n1+n2-(1-alpha)*n3)*1/(1+math.pow (math.e,-threshold))*(1-1/(1+math.pow(math.e,-threshold)))

def threshold_combine(acc_re_before,out_before,confidence,threshold_before,alpha):
    iter_num=500
    eta=0.002
    threshold_new=[]
    for i in range(len(threshold_before)):
        eps=-math.log(1/float(threshold_before[i])-1)
        for j in range(iter_num):
            # print(1/(1+math.pow(math.e,-eps)))
            eps=eps-eta*df_combine_acc(acc_re_before,out_before,confidence,i,eps,alpha)
        threshold_new.append(1/(1+math.pow(math.e,-eps)))
    return threshold_new

def get_acc_re_before(out_before,confidence_before,threshold_before):
    acc_re_before=[]
    for i in range(len(out_before)):
        if(confidence_before[i]>threshold_before[int(out_before[i])]):
            acc_re_before.append(0)
        else:
            acc_re_before.append(1)
    return acc_re_before

def threshold_update(dataset_dir,dataset_dir_threshold,alpha):

    # getting threshold_before
    threshold_file = open(os.path.join(dataset_dir_threshold, 'threshold_before.txt'))
    threshold_before = [i[:].strip('\n') for i in threshold_file.readlines()]

    # new seed -- LOTClass
    os.system('python src/train.py --dataset_dir '+dataset_dir_threshold)
    
    # getting confidence label
    #'out_before.txt' 'confidence_before.txt'  'out.txt' 'confidence.txt' 
    confidence_before_file = open(os.path.join(dataset_dir_threshold, 'confidence_before.txt'))
    confidence_before = [i[:].strip('\n') for i in confidence_before_file.readlines()]
    confidence_before_file.close()
    out_before_file = open(os.path.join(dataset_dir_threshold, 'out_before.txt'))
    out_before = [i[:].strip('\n') for i in out_before_file.readlines()]
    out_before_file.close()    
    confidence_file = open(os.path.join(dataset_dir_threshold, 'confidence.txt'))
    confidence = [i[:].strip('\n') for i in confidence_file.readlines()]
    confidence_file.close()

    # before acc-0/re-1 acc_re_before 
    acc_re_before=get_acc_re_before(out_before,confidence_before,threshold_before)

    # open
    # threshold_open_new=threshold_open(acc_re_before,out_before,confidence,threshold_before)
    # print(threshold_open_new)

    # accuracy
    # threshold_acc_new=threshold_acc(acc_re_before,out_before,confidence,threshold_before)
    # print(threshold_acc_new)

    # combine
    threshold_combine_new=threshold_combine(acc_re_before,out_before,confidence,threshold_before,alpha)
    # print(threshold_combine_new)

    #write
    f_out = open(os.path.join(dataset_dir, 'threshold.txt'),'w')
    for thre in threshold_combine_new:
        f_out.write(str(thre)+'\n')
    f_out.close()
    

