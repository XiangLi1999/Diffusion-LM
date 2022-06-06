import torch
import json
from transformers import BertForMaskedLM, BertTokenizer
filename = 'diffusion_lm/improved-diffusion/anlg_results/diff_roc_mbr.json2'
bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertForMaskedLM.from_pretrained(bert_model).cuda()

full_lst = []
with open(filename, 'r') as f:
    for line in f:
        line = json.loads(line)
        full_lst.append(line)

for example in full_lst:
    sent = example['sample']
    obs1 = example['obs1']
    obs2 = example['obs2']
    if 'UNK' in sent:
        sent = obs1 + sent.replace('UNK', tokenizer.mask_token) + obs2
        print(sent)
        model_inputs = tokenizer(sent,return_tensors="pt")
        model_inputs = {k:v.to(model.device) for k,v in model_inputs.items()}
        model_out = model(**model_inputs)
        mask_words = model_inputs['input_ids'] == tokenizer.mask_token_id
        masked_logits = model_out.logits[mask_words].view(-1, model_out.logits.size(-1))
        if masked_logits.size(0) > 0:
            # take argmax from this.
            max_cands = torch.max(masked_logits, dim=-1)
            indices = max_cands.indices
        model_inputs['input_ids'][mask_words] = indices
        print(tokenizer.batch_decode(model_inputs['input_ids'].tolist()))
    else:
        print('NO NEED THIS FIX. ')

