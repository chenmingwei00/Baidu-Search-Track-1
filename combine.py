base_dir='/home/aistudio/work/'
with open(base_dir+'ernie_dqa_task1_ner/applications/tasks/sequence_labeling/output/subtask1_test_pred_ner.txt','r') as pre1, \
        open(base_dir+'ernie_dqa_task1_ner_crf/applications/tasks/sequence_labeling/output/subtask1_test_pred_ner_crf.txt', 'r') as pre2, \
        open('./subtask1_test_pred.txt','w',encoding='utf-8') as pre_new,\
        open(base_dir+'ernie_dqa_task1_seg_attention/applications/tasks/sequence_labeling/output/subtask1_test_pred_seg.txt', 'r') as pre3:
    line1=pre1.readlines()
    line2=pre2.readlines()
    line3=pre3.readlines()

    for li1,li2,li3 in zip(line1,line2,line3):
        score1,ans1=li1.strip().split('\t')
        score2,ans2=li2.strip().split('\t')
        score3,ans3=li3.strip().split('\t')

        if ans1!='NoAnswer' and ans2=='NoAnswer' and ans3=='NoAnswer':
            pre_new.write(li2)
            continue
        elif ans1=='NoAnswer' and ans2!='NoAnswer' and ans3=='NoAnswer':
            pre_new.write(li1)
            continue
        elif ans1 == 'NoAnswer' and ans2 == 'NoAnswer' and ans3 != 'NoAnswer':
            pre_new.write(li1)
            continue
        elif ans1 == 'NoAnswer' and ans2 != 'NoAnswer' and ans3 != 'NoAnswer':
            flag2 = True
            flag3 = True
            comm = 0
            for char in ans2:
                if char in ans3:
                    comm += 1
            if comm / len(ans2) > 0.9:
                flag2 = False
            if comm / len(ans3) > 0.9 and flag2!=False:
                flag3 = False

            line = str(score1) + '\t'
            if flag2 == True: line += ans2 + ','
            if flag3 == True: line += ans3 + ','
            line += '\n'
            pre_new.write(line)
            continue
        elif ans1 != 'NoAnswer' and ans2 == 'NoAnswer' and ans3 != 'NoAnswer':
            flag1 = True
            flag3 = True
            comm = 0
            for char in ans1:
                if char in ans3:
                    comm += 1
            if comm / len(ans1) > 0.9:
                flag1 = False
            if comm / len(ans3) > 0.9 and flag1!=False:
                flag3 = False

            line = str(score1) + '\t'
            if flag1 == True: line += ans1 + ','
            if flag3 == True: line += ans3 + ','
            line += '\n'
            pre_new.write(line)
            continue
        elif ans1 != 'NoAnswer' and ans2 != 'NoAnswer' and ans3 == 'NoAnswer':
            flag1 = True
            flag2 = True
            comm = 0
            for char in ans1:
                if char in ans2:
                    comm += 1
            if comm / len(ans1) > 0.9:
                flag1 = False
            if comm / len(ans2) > 0.9 and flag1!=False:
                flag2 = False

            line = str(score1) + '\t'
            if flag1 == True:
                line += ans1 + ','
            if flag2 == True:
                line += ans2 + ','
            line += '\n'
            pre_new.write(line)
            continue
        elif ans1 != 'NoAnswer' and ans2 != 'NoAnswer' and ans3 != 'NoAnswer':
            flag1=True
            flag2=True
            flag3=True

            comm=0
            for char in ans1:
                if char in ans2 or char in ans3:
                    comm+=1
            if comm/len(ans1)>0.9:
                flag1=False

            comm = 0
            for char in ans2:
                if char in ans1 or char in ans3:
                    comm += 1
            if comm / len(ans2) > 0.9:
                flag2 = False

            comm = 0
            for char in ans3:
                if char in ans1 or char in ans2:
                    comm += 1
            if comm / len(ans3) > 0.9:
                flag3 = False
            line=str(score1)+'\t'
            if flag1==True:line+=ans1+','
            if flag2==True:line+=ans2+','
            if flag3==True:line+=ans3+','
            if flag1==False and flag2==False and flag3==False:
                len1=len(ans1)
                lem2=len(ans2)
                len3=len(ans3)
                lengths=[len1,lem2,len3]
                max_len=max(lengths)
                indx=lengths.index(max_len)
                all_answers=[ans1,ans2,ans3]
                line+=all_answers[indx]
            line+='\n'
            pre_new.write(line)
            continue
        else:
            line=str(score1)+'\t'+'NoAnswer'+'\n'
            pre_new.write(line)

print('finished combine submit files..............................')
"""

通过模型融合达到一定的提高，但提高有限，模型1 是crf  对应位置是STI比赛第一赛道任务一基线_副本 
ernie_dqa_task1_ner_crf  模型2 是ernie_dqa_task1_seg_attention  是错误的，再次训练查看
其效果，以及模型3dfdv 的ernie_dqa_task1_seg_attention
0.6787
"""
