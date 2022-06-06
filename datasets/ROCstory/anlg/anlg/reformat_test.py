import json
from collections import defaultdict
full_dict = defaultdict(list)
with open('test_gold_reference.txt', 'r') as f:
    for line in f:
        line = json.loads(line)
        # print(line.keys())
        assert line['label'] in ['2', '1']
        gold_ref = line['hyp1'] if line['label'] == '1' else line['hyp2']
        id2ref = {'story_id': line['story_id'], 'gold_ref':gold_ref} #'obs1': line['obs1'], 'obs2':line['obs2'],
        full_dict[line['story_id']].append(gold_ref)
print(full_dict.keys())
print(len(full_dict))

with open('test_cleanup_ref.json', 'w') as f:
    json.dump(full_dict, f)

