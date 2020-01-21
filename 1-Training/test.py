import train
import pandas as pd

# with open(filename, 'r') as f:
#     for line in f.readlines():
#         record = line.rstrip().split(',')
#         features = encode_example(
#             record, tokenizer, max_seq_length, labels_map)
#         yield ({'input_ids': features['input_ids'],
#                 'attention_mask': features['attention_mask'],
#                 'token_type_ids': features['token_type_ids']},
#                features['label'])


def get_data(filename):
    train = pd.read_csv(
        '/mnt/c/Users/phill/azure-ml/bert-stack-overflow/1-Training/data-shared-inbox/train.csv')

    with pd.read_csv(filename) as f:
        for line in f.iterrows():
            features = encode_example(
                record, tokenizer, max_seq_length, labels_map)
            yield ({'input_ids': features['input_ids'],
                    'attention_mask': features['attention_mask'],
                    'token_type_ids': features['token_type_ids']},
                   features['label'])


data = get_data(
    '/mnt/c/Users/phill/azure-ml/bert-stack-overflow/1-Training/data-shared-inbox/train.csv')
print(data)
