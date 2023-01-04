

import json
def test_data_split():
    """
    首先对预测文本doc_text,按照最大文本512 进行分割，分别预测，最后答案进行合并
    512-3-len(query) 就是每一个截断最大doc的长度
    """
    max_seq_len=512-3
    with open('./test_data/test.json','r',encoding='utf-8') as train_file,\
            open('./test_data/test_new.json','w',encoding='utf-8') as pre_del_file:
        lines=train_file.readlines()
        id=0
        for line in lines:
            t_data=json.loads(line)
            query=t_data['query']
            doc_text=t_data['doc_text']
            if len(query)+len(doc_text)>max_seq_len:
                doc_len=(max_seq_len-len(query))*2# 由于目前模型可以由2倍的长度
                new_docs=[]
                for k in range(len(doc_text)):
                    if len(new_docs)<doc_len:
                        new_docs.append(doc_text[k])
                    else:
                        t_data['doc_text']=''.join(new_docs)
                        t_data['id'] = id
                        t_data_line = json.dumps(t_data, ensure_ascii=False)
                        pre_del_file.write(t_data_line + '\n')
                        new_docs=[]
                t_data['doc_text'] = ''.join(new_docs)
                t_data['id'] = id
                t_data_line = json.dumps(t_data, ensure_ascii=False)
                pre_del_file.write(t_data_line + '\n')
                new_docs = []

            else:
                t_data['id']=id
                t_data_line=json.dumps(t_data,ensure_ascii=False)
                pre_del_file.write(t_data_line+'\n')
            id+=1
import  math

def test_move_split():
    """
    首先对预测文本doc_text,按照最大文本512,按照最大500，
    中间间隔250欢动窗口进行分割，然后把提取的答案按照id进行合并
    """
    max_seq_len=508
    with open('./test_data/test.json','r',encoding='utf-8') as train_file,\
            open('./test_data/test_seg.json','w',encoding='utf-8') as pre_del_file:
        lines=train_file.readlines()
        id=0
        for line in lines:
            t_data=json.loads(line)
            query=t_data['query']
            doc_text=t_data['doc_text']
            doc_ids=0
            if len(query)+len(doc_text)>max_seq_len and len(doc_text)>550:
                start=0
                end=max_seq_len-len(query)

                one_doctext=doc_text[start:end]
                t_data['doc_text'] = one_doctext
                t_data['id'] = id
                t_data["doc_ids"]=doc_ids
                t_data_line = json.dumps(t_data, ensure_ascii=False)
                pre_del_file.write(t_data_line + '\n')

                move_window=250
                move_step=math.ceil((len(doc_text)-end)/move_window)
                for k in range(move_step):
                    start+=move_window
                    end+=move_window
                    doc_ids+=1
                    new_docs=doc_text[start:end]
                    t_data['doc_text']=new_docs
                    t_data['id'] = id
                    t_data['doc_ids']=doc_ids
                    t_data_line = json.dumps(t_data, ensure_ascii=False)
                    pre_del_file.write(t_data_line + '\n')
            else:
                t_data['id']=id
                t_data['doc_ids']=doc_ids
                t_data_line=json.dumps(t_data,ensure_ascii=False)
                pre_del_file.write(t_data_line+'\n')
            id+=1
from tqdm import tqdm
def train_data_split():
    """
    对训练数据query+doc_text长度的文本进行截断形成多个数据集
    """
    max_seq_len=505
    with open('./train_data/train.json','r',encoding='utf-8') as train_file,\
            open('./train_data/train_split.json','w',encoding='utf-8') as pre_del_file:
        lines = train_file.readlines()
        id = 0

        new_lines=0
        for line in tqdm(lines):
            t_data = json.loads(line)
            query = t_data['query']
            doc_text = t_data['doc_text']
            answer_list = t_data['answer_list']
            answer_start_list=t_data['answer_start_list']
            org_answer=t_data['org_answer']

            doc_text=doc_text.replace('~',"#")
            #首先对答案进行扩充到与doc_text长度相同
            new_all_answer=[]
            new_answers_type=[]
            for start_indx,t_ans in zip(answer_start_list,answer_list):
                t_ans=t_ans.replace('~',"#")
                if len(new_all_answer)==0:
                    padd_ans=['~']*start_indx+list(t_ans)
                    padd_type=[0]*start_indx
                    padd_type+=[1]*len(list(t_ans))

                    new_answers_type.extend(padd_type)
                    new_all_answer.extend(padd_ans)
                elif len(new_all_answer)<=start_indx:#说后边答案与前边答案有距离
                    padd_len=start_indx-len(new_all_answer)
                    padd_ans=['~']*padd_len+list(t_ans)

                    padd_type = [0] * padd_len
                    padd_type += [1] * len(list(t_ans))

                    new_answers_type+=padd_type
                    new_all_answer+=padd_ans
            if len(new_all_answer)<len(list(doc_text)):
                new_all_answer+=['~']*(len(list(doc_text))-len(new_all_answer))
                new_answers_type+=[0]*(len(list(doc_text))-len(new_answers_type))

            assert len(new_all_answer)==len(doc_text)
            assert len(new_answers_type)==len(doc_text)

            if len(query) + len(doc_text) > max_seq_len:
                doc_len = max_seq_len - len(query)

                new_docs = []
                tmp_ans=[]
                tmp_ans_types=[]

                for k in range(len(doc_text)):

                    if len(new_docs) < doc_len:
                        new_docs.append(doc_text[k])
                        tmp_ans.append(new_all_answer[k])
                        tmp_ans_types.append(new_answers_type[k])
                    else:
                        new_traindata={}
                        new_answer_list = []
                        new_answer_start_list = []
                        new_org_answer = ''
                        first=True
                        tmp_constru_ans=''
                        for step,type_id in enumerate(tmp_ans_types):
                            if type_id==1 and first==True:
                                new_answer_start_list.append(step)
                                tmp_constru_ans+=tmp_ans[step]
                                first=False
                            elif type_id==1 and first==False:
                                tmp_constru_ans += tmp_ans[step]
                                first = False
                            elif type_id==0 and first==False:
                                new_answer_list.append(tmp_constru_ans)
                                tmp_constru_ans=''
                                first=True
                        if len(new_answer_list)==0:
                            new_org_answer='NoAnswer'
                        new_traindata['answer_list']=new_answer_list
                        new_traindata['answer_start_list']=new_answer_start_list
                        new_traindata['doc_text']=''.join(new_docs)
                        new_traindata['org_answer']=new_org_answer
                        new_traindata['query'] = t_data['query']
                        new_traindata['title'] = t_data['title']
                        new_traindata['url'] = t_data['url']

                        new_traindata['id'] = id
                        t_data_line = json.dumps(new_traindata, ensure_ascii=False)
                        pre_del_file.write(t_data_line + '\n')

                        new_docs = []
                        tmp_ans=[]
                        tmp_ans_types = []
                        new_lines+=1

                new_traindata = {}
                new_answer_list = []
                new_answer_start_list = []
                new_org_answer = ''
                first = True
                tmp_constru_ans = ''
                if sum(tmp_ans_types)>0:
                    for step, type_id in enumerate(tmp_ans_types):
                        if type_id == 1 and first == True:
                            new_answer_start_list.append(step)
                            tmp_constru_ans += tmp_ans[step]
                            first = False
                        elif type_id == 1 and first == False:
                            tmp_constru_ans += tmp_ans[step]
                            first = False
                        elif type_id == 0 and first == False:
                            new_answer_list.append(tmp_constru_ans)
                            tmp_constru_ans = ''
                            first = True
                    if len(new_answer_list) == 0:
                        new_org_answer = 'NoAnswer'
                    new_traindata['answer_list'] = new_answer_list
                    new_traindata['answer_start_list'] = new_answer_start_list
                    new_traindata['doc_text'] = ''.join(new_docs)
                    new_traindata['org_answer'] = new_org_answer
                    new_traindata['query'] = t_data['query']
                    new_traindata['title'] = t_data['title']
                    new_traindata['url'] = t_data['url']

                    new_traindata['id'] = id
                    t_data_line = json.dumps(new_traindata, ensure_ascii=False)
                    pre_del_file.write(t_data_line + '\n')

                    new_docs = []
                    tmp_ans = []
                    tmp_ans_types = []
                    new_lines += 1

            else:
                t_data['id'] = id
                t_data_line = json.dumps(t_data, ensure_ascii=False)
                pre_del_file.write(t_data_line + '\n')
                new_lines += 1

            id += 1
    print('finished.......................')
    print(new_lines)

def train_data_ner():
    """
    对训练数据query+doc_text长度的文本进行截断形成多个数据集
    """

    B_ans='B_ans'
    I_ans='I_ans'
    E_ans='E_ans'

    with open('./train_data/train.json','r',encoding='utf-8') as train_file,\
            open('./train_data/train_split.json','w',encoding='utf-8') as pre_del_file:
        lines = train_file.readlines()
        id = 0

        new_lines=0
        with open('./train.txt','w',encoding='utf-8') as train_tensorflow:
            for line in tqdm(lines):
                t_data = json.loads(line)
                query = t_data['query']
                doc_text = t_data['doc_text']
                answer_list = t_data['answer_list']
                answer_start_list=t_data['answer_start_list']
                org_answer=t_data['org_answer']
                # if len(answer_list)>1:
                #     print(org_answer)

                labels=['O']*len(doc_text)
                assert len(labels)==len(doc_text)
                for t_ans,t_start_indx in zip(answer_list,answer_start_list):
                    labels[t_start_indx]=B_ans
                    for k in range(t_start_indx+1,t_start_indx+len(t_ans)-1):
                        labels[k]=I_ans
                    labels[t_start_indx+len(t_ans)-1]=E_ans

                for char,lab in zip(doc_text,labels):
                    train_tensorflow.write(char+'\t'+lab+'\n')
                train_tensorflow.write('finished'+'\n')
                t_data['label']='\t'.join(labels)

                new_t_data=json.dumps(t_data,ensure_ascii=False)
                pre_del_file.write(new_t_data+'\n')

    print('finished.......................')
    print(new_lines)
import math
import math
def predict_deal():
    with open('./test_data/40001.txt', 'r', encoding='utf-8') as pre_answer, \
            open('./test_data/test_new.json', 'r', encoding='utf-8') as ori_json, \
            open('./test_data/subtask1_test_pred.txt', 'w', encoding='utf-8') as final_pre:
        pre_lines = pre_answer.readlines()
        input_lines = ori_json.readlines()

        assert len(pre_lines) == len(input_lines)
        id2answer = {}
        for t_pre, t_input in zip(pre_lines, input_lines):
            # t_pre=t_pre.replace('sep]','')
            t_inputs = json.loads(t_input)
            if t_inputs['id'] not in id2answer:
                id2answer[t_inputs['id']] = [t_pre]
            else:
                id2answer[t_inputs['id']].append(t_pre)
        id2answer_list = sorted(id2answer.items(), key=lambda k: k[0])
        for id, ansers_t in id2answer_list:
            trues_answer = []
            true_answer_score = []
            no_answer = []
            no_answer_scores = []
            for t_ans in ansers_t:
                t_ans = t_ans.strip().split('\t')
                if t_ans[1] == 'NoAnswer' or t_ans[1]=="sep]" or t_ans[1]=="[sep]":
                    no_answer_scores.append(float(t_ans[0]))
                    no_answer.append(t_ans[1])
                else:
                    t_pre = t_ans[1].replace('sep]', '')
                    if len(t_pre) > 1:
                        trues_answer.append(t_pre)
                        true_answer_score.append(float(t_ans[0]))
            if len(trues_answer) <= 0:
                try:
                    score = sum(no_answer_scores) / len(no_answer_scores)
                    line = str(score) + '\t' + 'NoAnswer' + '\n'
                except:
                    score = 0.9
                    line = str(score) + '\t' + 'NoAnswer' + '\n'
            else:
                score = sum(true_answer_score) / len(true_answer_score)
                line = str(score) + '\t' + ';'.join(trues_answer) + '\n'
                line = line.replace('#', '')
            final_pre.write(line)

import copy
from tqdm import tqdm
def train_nearest_context():
    """
    对doc_text进行就近上下文截断策略，形成pre_context+ans的doc_text ,总长度<=512,其中
    pre_context>=100,ans<=400。
    基于以上数据形成新的doc_text,ori_ans ,以及答案的新的开始位置
    :return:
    """
    ori_train_path='./train_data/train_split.json'
    new_train_path='./train_data/train_split_2.json'
    max_doc_len=500

    with open(ori_train_path,'r',encoding='utf-8') as files_ori, \
            open(new_train_path, 'w', encoding='utf-8') as files_new:
        lines=files_ori.readlines()
        for line in tqdm(lines):
            t_data=json.loads(line)
            answer_list=t_data['answer_list']
            answer_start_list=t_data['answer_start_list']
            doc_text = t_data['doc_text']
            if len(doc_text)>max_doc_len:
                if len(answer_start_list)>1:
                    one_start=answer_start_list[0]
                    final_start=answer_start_list[-1]

                    final_index=len(answer_list[-1])
                    final_end=final_start+final_index
                    if final_end-one_start<max_doc_len:#说明所有答案加上中间位置小于最大长度
                        label=t_data['label'].strip().split('\t')
                        assert len(label)==len(doc_text)
                        new_label=label[one_start:final_end]
                        new_doc_text=list(doc_text[one_start:final_end])

                        pre_context=one_start-1
                        one_newstart_indx=0
                        answer_start_dis=[answer_start_list[ele]-one_start for ele in range(1,len(answer_start_list))]
                        answer_start_dis.insert(0,0)
                        rest_lenght=(max_doc_len-len(new_label))/4 #

                        while len(new_label)<max_doc_len and pre_context>0 and one_newstart_indx<=rest_lenght*3:
                            new_label.insert(0,label[pre_context])
                            new_doc_text.insert(0,doc_text[pre_context])
                            one_newstart_indx+=1
                            pre_context-=1
                        post_index=final_end
                        answer_start_ids=[ele+one_newstart_indx for ele in answer_start_dis]
                        while len(new_label) < max_doc_len and post_index < len(doc_text):
                            new_label.append(label[post_index])
                            new_doc_text.append(doc_text[post_index])
                            post_index += 1
                        new_label='\t'.join(new_label)
                        new_doc_text=''.join(new_doc_text)

                        t_data['answer_start_list']=answer_start_ids
                        t_data['label']=new_label
                        t_data['doc_text']=new_doc_text
                        new_t_data_line = json.dumps(t_data, ensure_ascii=False)
                        files_new.write(new_t_data_line + '\n')
                    else:# 如果答案之间的总长度大于最大限制
                        seg_answer_starts=[]
                        seg_answer_texdts=[]
                        move_start=0

                        while move_start<len(answer_start_list):
                            current_start_index = [one_start]
                            current_answer_texts = [answer_list[move_start]]

                            if move_start+1<len(answer_start_list):
                                second_end_indx = answer_start_list[move_start + 1] + len(answer_list[move_start + 1])
                                one_start = answer_start_list[move_start]
                                while second_end_indx-one_start<max_doc_len:
                                    current_start_index.append(answer_start_list[move_start + 1])
                                    current_answer_texts.append(answer_list[move_start+1])
                                    move_start+=1
                                    if move_start+1>=len(answer_start_list):
                                        break
                                    second_end_indx = answer_start_list[move_start + 1] + len(answer_list[move_start + 1])

                            seg_answer_starts.append(current_start_index)
                            seg_answer_texdts.append(current_answer_texts)
                            move_start+=1
                        for t_anser_start,t_ans in zip(seg_answer_starts,seg_answer_texdts):

                            one_start=t_anser_start[0]
                            final_start = t_anser_start[-1]
                            final_index = len(t_ans[-1])
                            final_end = final_start + final_index

                            label = t_data['label'].strip().split('\t')
                            assert len(label) == len(doc_text)
                            new_label = label[one_start:final_end]

                            new_doc_text = list(doc_text[one_start:final_end])

                            pre_context = one_start - 1
                            one_newstart_indx = 0
                            answer_start_dis = [t_anser_start[ele] - one_start for ele in
                                                range(1, len(t_anser_start))]
                            answer_start_dis.insert(0, 0)
                            rest_lenght = (max_doc_len - len(new_label)) / 4  #

                            while len(new_label) < max_doc_len and pre_context > 0 and one_newstart_indx<=rest_lenght*3:
                                new_label.insert(0, label[pre_context])
                                new_doc_text.insert(0, doc_text[pre_context])
                                one_newstart_indx += 1
                                pre_context -= 1
                            post_index = final_end
                            answer_start_ids = [ele + one_newstart_indx for ele in answer_start_dis]
                            while len(new_label) < max_doc_len and post_index < len(doc_text):
                                new_label.append(label[post_index])
                                new_doc_text.append(doc_text[post_index])
                                post_index += 1
                            new_label = '\t'.join(new_label)
                            new_doc_text = ''.join(new_doc_text)
                            new_t_data=copy.deepcopy(t_data)
                            new_t_data['answer_start_list'] = answer_start_ids
                            new_t_data['label'] = new_label
                            new_t_data['doc_text'] = new_doc_text
                            new_t_data['answer_list']=t_ans
                            new_t_data_line = json.dumps(new_t_data, ensure_ascii=False)
                            files_new.write(new_t_data_line + '\n')
                elif len(answer_start_list)==1:
                    #如果所有答案只有一个
                    one_start=answer_start_list[0]
                    final_end=len(answer_list[0])+one_start
                    if final_end-one_start<max_doc_len: #答案小于最大长度
                        label = t_data['label'].strip().split('\t')
                        assert len(label) == len(doc_text)
                        new_label = label[one_start:final_end]

                        new_doc_text = list(doc_text[one_start:final_end])

                        pre_context = one_start - 1
                        one_newstart_indx = 0

                        rest_lenght = (max_doc_len - len(new_label)) / 4  #
                        answer_start_dis = [answer_start_list[ele] - one_start for ele in
                                            range(len(answer_start_list))]
                        while len(new_label) < max_doc_len and pre_context >=0 and one_newstart_indx <= rest_lenght * 3:
                            new_label.insert(0, label[pre_context])
                            new_doc_text.insert(0, doc_text[pre_context])
                            one_newstart_indx += 1
                            pre_context -= 1
                        post_index = final_end
                        answer_start_ids = [ele + one_newstart_indx for ele in answer_start_dis]
                        while len(new_label) < max_doc_len and post_index < len(doc_text):
                            new_label.append(label[post_index])
                            new_doc_text.append(doc_text[post_index])
                            post_index += 1
                        new_label = '\t'.join(new_label)
                        new_doc_text = ''.join(new_doc_text)
                        new_t_data = copy.deepcopy(t_data)
                        new_t_data['answer_start_list'] = answer_start_ids
                        new_t_data['label'] = new_label
                        new_t_data['doc_text'] = new_doc_text
                        new_t_data_line=json.dumps(new_t_data,ensure_ascii=False)
                        files_new.write(new_t_data_line+'\n')

                    else:
                        #如果答案本身唱过限制最大长度
                        files_new.write(line)
                else:
                    #如果没有答案
                    files_new.write(line)
            else:
                #如果doctext本身长度小于限制最大长度，则直接满足
                files_new.write(line)
def train_data_split2seg():
    """
    首先分为两个段落，最大长度为505，对应的anserstart以及answerlist分为两个部分
    :return:
    """
    new_train_path = './train_data/train_split.json'
    train_path = './train_data/train_split_2.json'

    max_doc_len = 505

    with open(new_train_path, 'r', encoding='utf-8') as files_ori, \
            open(train_path, 'w', encoding='utf-8') as files_new:

        lines=files_ori.readlines()
        for lin in lines:
            t_data=json.loads(lin)
            doc_text=t_data['doc_text']
            answer_list=t_data['answer_list']
            answer_start_list=t_data['answer_start_list']
            label=t_data['label'].strip().split('\t')
            org_answer=t_data['org_answer']

            if len(doc_text)>max_doc_len:
                print()


if __name__ == '__main__':
    # test_data_split()
    predict_deal()
    # train_data_split()
    # train_data_ner()
    # train_data_split2seg()
    # train_nearest_context()

    # with open('./train_data/train.json','r',encoding='utf-8') as files_read:
    #     lines=files_read.readlines()
    #     print(len(lines))
    #
    #     no_answer=0
    #     for line in lines:
    #          data=json.loads(line)
    #          ansecout=data['answer_start_list']
    #          if len(ansecout)<=0:
    #              no_answer+=1
    #     print(no_answer)
