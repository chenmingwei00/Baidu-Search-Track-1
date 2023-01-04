import json
import matplotlib.pyplot as plt
all_data_counts=0
max_data=0
query_len={}
doc_leng={}

all_answer_count=0
over_max_count=0
with open('./train_data/train.json','r',encoding='utf-8') as train_file:
    lines=train_file.readlines()
    for line in lines:

        t_data=json.loads(line)
        query=t_data['query']
        doc_text=t_data['doc_text']
        if len(query)+len(doc_text)>512:
            max_data+=1

        if len(query) not in query_len:
            query_len[len(query)]=1
        else:
            query_len[len(query)] += 1

        if len(doc_text) not in doc_leng:
            if len(doc_text)>900:
                doc_leng[900]=1
            else:
                doc_leng[len(doc_text)] = 1
        else:
            if len(doc_text)>900:
                doc_leng[900]+=1
            else:
                doc_leng[len(doc_text)] += 1
        answer_start_list=t_data['answer_start_list']
        answer_list=t_data['answer_list']

        for ans,start_indx in zip(answer_list,answer_start_list):
            final_indx=start_indx+len(ans)
            if final_indx>512:
                over_max_count+=1
            all_answer_count+=1


print(max_data/len(lines))
print(over_max_count/all_answer_count)
print('11111')
doc_leng_list=sorted(doc_leng.items(),key=lambda k:k[0])
for k,v in doc_leng_list:
    plt.bar(k,v)
plt.show()