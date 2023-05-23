import math
import os
import nltk
from nltk.stem.lancaster import LancasterStemmer

# delete .pt
def del_pt(dataset_dir):
    if(os.path.exists(dataset_dir+'final_model.pt')):
        os.remove(dataset_dir+'final_model.pt')
    if(os.path.exists(dataset_dir+'category_vocab.pt')):
        os.remove(dataset_dir+'category_vocab.pt') 
    if(os.path.exists(dataset_dir+'mcp_train.pt')):
        os.remove(dataset_dir+'mcp_train.pt')
    if(os.path.exists(dataset_dir+'mcp_model.pt')):
        os.remove(dataset_dir+'mcp_model.pt')
    if(os.path.exists(dataset_dir+'label_name_data.pt')):
        os.remove(dataset_dir+'label_name_data.pt')

# getting seed words candidate set 'category_vocab.txt'
def get_category_vocab(dataset_dir,cv_name):
    category_vocab_file = open(os.path.join(dataset_dir, cv_name))
    category_vocab = [i[:-1].strip('[]').split(', ') for i in category_vocab_file.readlines()]
    category_vocab_file.close()
    return category_vocab

# getting seed words 'label_names.txt'
def get_label_delete (dataset_dir,label_name):
    label_names_file = open(os.path.join(dataset_dir, label_name))
    label_delete = [i[:].strip('\n').split(' ') for i in label_names_file.readlines()]
    label_names_file.close()
    return label_delete

def copy_txt(dataset_dir,name1,name2):
    fp = open(os.path.join(dataset_dir,name1),'r')
    labels = [i[:].strip('\n').split(' ') for i in fp.readlines()]
    fq = open(os.path.join(dataset_dir, name2),'w')
    for label in labels:
        fq.write(' '.join(label) + '\n')
    fp.close()
    fq.close()

def copy_txt_2(dataset_dir,name1,dataset_dir2,name2):
    fp = open(os.path.join(dataset_dir,name1),'r',encoding='utf-8')
    labels = [i[:].strip('\n').split(' ') for i in fp.readlines()]
    fq = open(os.path.join(dataset_dir2, name2),'w',encoding='utf-8')
    for label in labels:
        fq.write(' '.join(label) + '\n')
    fp.close()
    fq.close()

# getting the new labe names after the seed word is added
def gene_label_new(dataset_dir,before_name,new_name, i, str_cv):
    #read label_name_before
    label_names_file = open(os.path.join(dataset_dir, before_name))
    label_old = label_names_file.readlines()
    label_new=[]
    for k in range(len(label_old)):
        if(k!=i):
            label_new.append(label_old[k])
        else:
            label_new.append(label_old[i].replace("\n", " ")+ str_cv + '\n')  #str_cv_lemma.replace("'", "")
    f_out = open(os.path.join(dataset_dir, new_name),'w')
    for label in label_new:
        f_out.write(str(label))
    f_out.close()

# getting the new labe names after the seed word is deleted
def gene_label_del(dataset_dir,before_name,new_name, i, str_cv):
    # read label_name_before
    label_names_file = open(os.path.join(dataset_dir, before_name))
    labels = [i[:].strip('\n').split(' ') for i in label_names_file.readlines()]
    labels[i].remove(str_cv)
    f_out = open(os.path.join(dataset_dir, new_name),'w')
    for label in labels:
        f_out.write(' '.join(label) + '\n')
    f_out.close()

# generaing path for dataset_dir_1,dataset_dir_2
def get_path(dataset_dir):
    strs=dataset_dir.split('_')
    time=strs[1].strip('/')
    dataset_dir_1=strs[0]+'_'+str(int(time)-1)+'/'
    dataset_dir_2=strs[0]+'_'+str(int(time)-2)+'/'
    return dataset_dir_1,dataset_dir_2

def get_map_list(dataset_dir):
    dataset_dir_1,dataset_dir_2=get_path(dataset_dir)

    map_list=[]
    for i in range(8):
        map_list.append({})

    performence_1_file = open(os.path.join(dataset_dir_1, 'seedDelete_performence.txt'))
    performence_1 = [i[:].strip('\n').split(' ') for i in performence_1_file.readlines()]
    performence_2_file = open(os.path.join(dataset_dir_2, 'seedDelete_performence.txt'))
    performence_2 = [i[:].strip('\n').split(' ') for i in performence_2_file.readlines()]

    for performence in performence_1:
        strs=performence[1].split(':')
        if(float(strs[1])>0):
            map_list[int(performence[0])][strs[0]]=1
    for performence in performence_2:
        strs=performence[1].split(':')
        if(float(strs[1])>0):
            if strs[0] in map_list[int(performence[0])]:
                map_list[int(performence[0])][strs[0]]=map_list[int(performence[0])][strs[0]]+1
            else:
                map_list[int(performence[0])][strs[0]]=1
    
    return map_list

def c_umass(top_words, documents):
    coherence_score = 0.0
    e = 0.0000000001
    for m in range(1,len(top_words)):
        for l in range(1,m):
            p_j = DocumentFrequency(documents, top_words[l]) / len(documents)
            p_i_j = DocumentFrequency2(documents, top_words[m], top_words[l]) / len(documents)
            if p_j !=0:
                coherence_score = coherence_score + math.log((p_i_j + e) / (p_j))
    coherence_score=coherence_score*2/(len(top_words)*(len(top_words)-1))
    return coherence_score

def DocumentFrequency(documents, word):
    count = 0
    for i in range(len(documents)):
        for j in range(len(documents[i])):
            if documents[i][j] == word:
                count=count+1
                break
    return count

def DocumentFrequency2(documents, word_i, word_j):
    count = 0
    for i in range(len(documents)):
        exsit_i = False
        exsit_j = False
        for j in range(len(documents[i])):
            if(documents[i][j] == word_i):
                exsit_i = True
                break
        for j in range(len(documents[i])):
            if(documents[i][j] == word_j):
                exsit_j = True
                break
        if (exsit_i and exsit_j):
            count=count+1
    return count

def coherence(dataset_dir,category_vocab_name,txt_name):
    category_vocab=get_category_vocab(dataset_dir,category_vocab_name)
    top_words=[]
    for cv in category_vocab:
        topic=[]
        if(len(cv)!=1):
            for word in cv:
                topic.append(word.split('\'')[1])
            top_words.append(topic)

    text_file = open(os.path.join(dataset_dir, txt_name), encoding="utf-8")
    doc= [doc.strip() for doc in text_file.readlines()]
    documents = []
    for sentence in doc:
        documents.append(list(sentence.split(' ')))

    sum_coherence=0
    for i in range(1,len(top_words)):
        sum_coherence=sum_coherence+c_umass(top_words[i], documents)

    return sum_coherence/len(top_words)

def seed_update(dataset_dir,start=0,num_seed=5,flag_delete=0):
    category_vocab=get_category_vocab(dataset_dir,'category_vocab_accept.txt')  #category_vocab_std.txt
    label_delete=get_label_delete (dataset_dir,'label_names.txt')
    label_names=get_label_delete (dataset_dir,'label_names.txt')
    # label_names --> label_name_cur
    copy_txt(dataset_dir,'label_names.txt','label_names_update.txt')
    # getting coherence_std
    coherence_std=coherence(dataset_dir,'category_vocab_std.txt','train.txt')
    print('coherence_std:'+str(coherence_std))

    # adding seed words
    f_out = open(os.path.join(dataset_dir, "seedUpdate_performence.txt"),'w')
    f_out.write('coherence_std:'+str(coherence_std)+ '\n')

    # Marks -- the position of the current candidate seed word
    flag=[]
    for i in range(len(category_vocab)):
        flag.append(start-1)

    for j in range(num_seed):
        for i in range(len(category_vocab)):
            if(len(category_vocab[i])==1):
                continue
            for k in range(flag[i]+1,len(category_vocab[i])): # Remove words with the same root as the existing seed words
                name_cur=category_vocab[i][k][1:-1]
                flag_cur=1
                for m in range(len(label_names[i])):
                    if(LancasterStemmer().stem(label_names[i][m])==LancasterStemmer().stem(name_cur)):
                        flag_cur=0
                if(flag_cur==1):
                    # print(name_cur)
                    flag[i]=k

                    update_file = open(os.path.join(dataset_dir, 'label_names_update.txt'))
                    update = [i[:].strip('\n').split(' ') for i in update_file .readlines()]
                    # generating label_name_new
                    gene_label_new(dataset_dir,'label_names_update.txt','label_names_cur.txt',i,name_cur)
                    # caculate --> 'category_vocab.txt'
                    del_pt(dataset_dir)
                    os.system('python src/train2.py --dataset_dir '+dataset_dir +' --label_names_file label_names_cur.txt')
                    # getting coherence_cur
                    coherence_cur=coherence(dataset_dir,'category_vocab.txt','train.txt')
                    if(coherence_cur>coherence_std):
                        coherence_std=coherence_cur
                        copy_txt(dataset_dir,'label_names_cur.txt','label_names_update.txt')
                    f_out.write(str(i)+' '+name_cur+':'+str(coherence_cur) + '\n')

                    break

    f_out.close()
    copy_txt(dataset_dir,'label_names_update.txt','label_names_addition.txt')

    # deleting seed words
    f_out = open(os.path.join(dataset_dir, "seedDelete_performence.txt"),'w')
    if(1==flag_delete):
        map_list=get_map_list(dataset_dir) # Evidence from the last two periods
    for i in range(len(label_delete)):
        for j in range(len(label_delete[i])):
            # generating label_name_new
            gene_label_del(dataset_dir,'label_names_update.txt','label_names_cur.txt',i,label_delete[i][j])
            # caculate -->  'category_vocab.txt'
            del_pt(dataset_dir)
            os.system('python src/train2.py --dataset_dir '+ dataset_dir +' --label_names_file label_names_cur.txt')
            # getting coherence_cur
            coherence_cur=coherence(dataset_dir,'category_vocab.txt','train.txt')
            f_out.write(str(i)+' '+label_delete[i][j]+':'+str(coherence_cur-coherence_std) + '\n')

            # If the category has only one seed word, no delete
            update_file = open(os.path.join(dataset_dir, 'label_names_update.txt'))
            update = [i[:].strip('\n').split(' ') for i in update_file .readlines()]
            if(1==len(update[i])):
                continue

            # decide whether to delete
            if(1==flag_delete):
                if label_delete[i][j] in map_list[i]:
                    if(coherence_cur>coherence_std and map_list[i][label_delete[i][j]]>=1): #map_list[i][label_delete[i][j]]>1
                        coherence_std=coherence_cur
                        copy_txt(dataset_dir,'label_names_cur.txt','label_names_update.txt')

            # if(coherence_cur>coherence_std):
            #     coherence_std=coherence_cur
            #     copy_txt(dataset_dir,'label_names_cur.txt','label_names_update.txt')

    f_out.close()
    copy_txt(dataset_dir,'label_names_update.txt','label_names_next.txt')