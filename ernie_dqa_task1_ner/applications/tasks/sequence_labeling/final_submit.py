import json
def predict_deal():
    with open('./output/4001.txt','r',encoding='utf-8') as pre_answer,\
        open('./data/test_data/test.json','r',encoding='utf-8') as ori_json,\
        open('./output/subtask1_test_pred_ner.txt','w',encoding='utf-8') as final_pre:
        pre_lines=pre_answer.readlines()
        input_lines=ori_json.readlines()

        assert len(pre_lines)==len(input_lines)
        id2answer={}
        for t_pre,t_input in zip(pre_lines,input_lines):
            # t_pre=t_pre.replace('[MASK]','')
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
                    # if len(t_ans[1])>:
                    trues_answer.append(t_ans[1])
                    true_answer_score.append(float(t_ans[0]))
            if len(trues_answer)<=0:
                try:
                    score=sum(no_answer_scores)/len(no_answer_scores)
                    line=str(score)+'\t'+'NoAnswer'+'\n'
                except:
                    score = 0.9
                    line = str(score) + '\t' + 'NoAnswer' + '\n'
            else:
                score=sum(true_answer_score)/len(true_answer_score)
                line=str(score)+'\t'+';'.join(trues_answer)+'\n'
                line=line.replace('#','')
            final_pre.write(line)
    print('finished construct submit files.............')
if __name__ == '__main__':
    predict_deal()
