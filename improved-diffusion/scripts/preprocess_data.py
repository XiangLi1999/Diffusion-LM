import json 
import csv
import random
random.seed(10)
sentence_lst = []
roc_train = 'diffusion_lm/ROCstory/ROCstory_full.csv'
with open(roc_train, 'r') as csvfile:
    roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
    for row in roc_reader:
        sentences = " ".join(row[2:])
        sentence_lst.append(sentences)
sentence_lst = sentence_lst[1:]
print(len(sentence_lst))
print(sentence_lst[:2])

# write to dev and test sets. --> 5k valid.
random.shuffle(sentence_lst)
print()
print(sentence_lst[:2])
valid_lst = sentence_lst[:5000]
train_lst = sentence_lst[5000:]

with open('diffusion_lm/ROCstory/roc_valid.json', 'w') as f:
    for sent in valid_lst:
        print(json.dumps([sent]), file=f)

with open('diffusion_lm/ROCstory/roc_train.json', 'w') as f:
    for sent in train_lst:
        print(json.dumps([sent]), file=f)




