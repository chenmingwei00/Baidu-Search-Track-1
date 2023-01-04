

import json
def test_data_split():
    """
    首先对预测文本doc_text,按照最大文本512 进行分割，分别预测，最后答案进行合并
    """
    max_seq_len=505
    with open('./test_data/test_pre.json','r',encoding='utf-8') as train_file,\
            open('./test_data/test.json','w',encoding='utf-8') as pre_del_file:
        lines=train_file.readlines()
        id=0
        for line in lines:
            t_data=json.loads(line)
            query=t_data['query']
            doc_text=t_data['doc_text']
            if len(query)+len(doc_text)>max_seq_len:
                doc_len=max_seq_len-len(query)
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
    with open('./test_data/4000.txt','r',encoding='utf-8') as pre_answer,\
        open('./test_data/test_new.json','r',encoding='utf-8') as ori_json,\
        open('./test_data/subtask1_test_pred.txt','w',encoding='utf-8') as final_pre:
        pre_lines=pre_answer.readlines()
        input_lines=ori_json.readlines()

        assert len(pre_lines)==len(input_lines)
        id2answer={}
        for t_pre,t_input in zip(pre_lines,input_lines):
            t_inputs=json.loads(t_input)
            if t_inputs['id'] not in id2answer:
                id2answer[t_inputs['id']]=[t_pre]
            else:
                id2answer[t_inputs['id']].append(t_pre)
        id2answer_list=sorted(id2answer.items(),key=lambda k:k[0])
        for id, ansers_t in id2answer_list:
            trues_answer = []
            true_answer_score = []
            no_answer = []
            no_answer_scores = []
            for t_ans in ansers_t:
                t_ans = t_ans.strip().split('\t')
                if t_ans[1] == 'NoAnswer':
                    no_answer_scores.append(float(t_ans[0]))
                    no_answer.append(t_ans[1])
                else:
                    trues_answer.append(t_ans[1])
                    true_answer_score.append(float(t_ans[0]))
            if len(trues_answer)<=0:
                score=sum(no_answer_scores)/len(no_answer_scores)
                line=str(score)+'\t'+'NoAnswer'+'\n'
            else:
                score=sum(true_answer_score)/len(true_answer_score)
                line=str(score)+'\t'+''.join(trues_answer)+'\n'
                line=line.replace('#','')
            final_pre.write(line)



if __name__ == '__main__':
    # train_data_split()
    train_data_ner()
    # predict_deal()


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
