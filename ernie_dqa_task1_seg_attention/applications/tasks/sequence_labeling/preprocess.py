import os
import sys
sys.path.append("../../../")
from erniekit.data.tokenizer.tokenization_wp import FullTokenizer
from collections import namedtuple
import json
from erniekit.utils.util_helper import truncation_words
from tqdm import tqdm
from erniekit.data.util_helper import pad_batch_data

def convert_texts_to_ids(examples, max_pos_id=512):
        src_ids = []
        position_ids = []
        sentence_ids = []
        labels_start = []
        labels_end = []

        tokenizer = FullTokenizer(
            vocab_file="../../models_hub/ernie_3.0_base_ch_dir/vocab.txt",
            split_char=" ",
            unk_token="[UNK]",
            params=None
        )

        for sample in tqdm(examples):
            query = sample['query']
            doc = sample['document']
            org_answer = sample['org_answer']
            answer_list = sample['answer_list']
            answer_start_list = sample['answer_start_list']

            tokens_query = tokenizer.tokenize(query)
            tokens_doc = tokenizer.tokenize(doc)

            # 加上截断策略
            if len(tokens_doc) > 512 - 3 - len(tokens_query):
                tokens_doc = truncation_words(tokens_doc, 512 - 3 - len(tokens_query), 0)

            sentence_id = []
            tokens = []
            tokens.append("[CLS]")
            sentence_id.append(0)
            for token in tokens_query:
                tokens.append(token)
                sentence_id.append(0)
            tokens.append("[SEP]")
            sentence_id.append(0)
            for token in tokens_doc:
                tokens.append(token)
                sentence_id.append(1)
            tokens.append("[SEP]")
            sentence_id.append(1)

            src_id = tokenizer.convert_tokens_to_ids(tokens)
            
            src_ids.append(src_id)
            pos_id = list(range(len(src_id)))
            position_ids.append(pos_id)
            sentence_ids.append(sentence_id)
            
            answer_span = []         
            label_start = [0.0] * len(src_id)
            label_end = [0.0] * len(src_id)
            if org_answer != "NoAnswer":
                for answer, answer_start in zip(answer_list, answer_start_list):
                    pre_doc = doc[:answer_start]
                    tokens_pre_doc = tokenizer.tokenize(pre_doc)
                    tokens_answer = tokenizer.tokenize(answer)
                    start_index = len(tokens_pre_doc) + len(tokens_query) + 2 if answer_start != 0 else len(tokens_query) + 2
                    end_index = start_index + len(tokens_answer) - 1
                    answer_span.append((start_index, end_index))
            else:
                answer_span.append((0, 0))
            for span in answer_span:
                if span[0] < len(label_start):
                    label_start[span[0]] = 1.0
                if span[1] < len(label_end): 
                    label_end[span[1]] = 1.0
            labels_start.append(label_start)
            labels_end.append(label_end)

        return_list_ids = []
        padded_ids, input_mask, batch_seq_lens = pad_batch_data(src_ids,
                                                                pad_idx=0,
                                                                return_input_mask=True,
                                                                return_seq_lens=True)
        sent_ids_batch = pad_batch_data(sentence_ids, pad_idx=0)
        pos_ids_batch = pad_batch_data(position_ids, pad_idx=0)

        return_list_ids.append(padded_ids)  # append src_ids  [-1, -1]
        return_list_ids.append(sent_ids_batch)  # append sent_ids  [-1, -1]
        return_list_ids.append(input_mask)  # append mask_ids   [-1, -1]
        return_list_ids.append(pos_ids_batch)  # append pos_ids   [-1, -1]
        return_list_ids.append(batch_seq_lens)  # append seq_lens   [-1]

        labels_start_batch = pad_batch_data(labels_start, pad_idx=-100)
        labels_end_batch = pad_batch_data(labels_end, pad_idx=-100)
        return_list_ids.append(labels_start_batch)   # start index  [-1, -1]
        return_list_ids.append(labels_end_batch)     # end index    [-1, -1]
        return return_list_ids


def read_json_files(file_path):
        line_index = 0
        with open(file_path, "r", encoding='utf-8') as f:
            examples = []
            for linenum, line in enumerate(f):
                line_index = linenum + 1
                data_dict = {}
                sample = json.loads(line.strip())
                data_dict['query'] = sample['query']
                data_dict['document'] = sample['doc_text']
                data_dict['org_answer'] = sample['org_answer']
                data_dict['answer_list'] = sample['answer_list']
                data_dict['answer_start_list']= sample['answer_start_list']
                examples.append(data_dict)

            return examples


if __name__ == '__main__':
    dataset = ['dev']
    for data in dataset:
        examples = read_json_files(f'data/{data}_data/{data}.json')
        src_ids, sent_ids, input_mask, pos_ids, seq_len, label_start, label_end = convert_texts_to_ids(examples)
        for src_id, sent_id, mask in zip(src_ids, sent_ids, input_mask):
            pass
