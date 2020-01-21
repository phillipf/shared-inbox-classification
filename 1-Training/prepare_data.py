import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle

emails = []
for line in open('file.json', 'r'):
    emails.append(json.loads(line))

df = pd.DataFrame.from_records(emails)
df.head()

labels = ['negative', 'neutral']

annotations = pd.read_csv('data.csv')
annotations['value'] = 1

#consider re-factoring to sklearn multilabel binarizer
# annotations = annotations.pivot_table('value', ['text'], 'label', fill_value=0).reset_index()

# annotations['labels'] = list(zip(annotations['reply required'].tolist(),
#                                  annotations['positive'].tolist(),
#                                  annotations['neutral'].tolist(),
#                                  annotations['negative'].tolist()
#                                  #annotations['no reply required'].tolist(), this is duplicated info
#                                  )
#                              )

# np.random.seed(100)
# msk = np.random.rand(len(annotations)) < 0.8

# train_df = annotations[msk][['text', 'labels']]
# validation_df = annotations[~msk][['text', 'labels']]

# train_df.to_csv('train.csv')

# shuffle data 
bd = shuffle(annotations)

# split data into train, test, and valid sets
msk = np.random.rand(len(bd)) < 0.7
train = bd[msk]
temp = bd[~msk]
msk = np.random.rand(len(temp)) < 0.66
valid = temp[msk]
test = temp[~msk]

output_dir = '/mnt/c/Users/phill/azure-ml/bert-stack-overflow/1-Training/data-shared-inbox'

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create and save classes.txt file
classes = pd.DataFrame(bd['label'].unique().tolist())
classes.to_csv(os.path.join(output_dir, 'classes.txt'), header=False, index=False)

# save train, valid, and test files
train.to_csv(os.path.join(output_dir, 'train.csv'), header=False, index=False)
valid.to_csv(os.path.join(output_dir, 'valid.csv'), header=False, index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), header=False, index=False)