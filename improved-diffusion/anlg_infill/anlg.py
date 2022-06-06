import json
import sys, os, torch
from spacy.lang.en import English
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from transformers import AutoModelForCausalLM
# read files.
# with open('diffusion_lm/ROCstory/anlg/anlg/dev_cleanup.json', 'r') as f:
SPLIT = 'test'

if SPLIT == 'val':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/dev_cleanup.json'
elif SPLIT == 'test':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/test_cleanup_no_label.json'
else:
    assert False, "invalid split"

with open(source_file, 'r') as f:
    sent_lst = json.load(f)


nlp = English()
tokenizer = nlp.tokenizer
MODE = 'ar'

'''
 "00b9adb2-b3b6-4737-902a-50f308bac4b5-1": {
        "gold_labels": [
            "I put my baby in the car and drove around.",
            "I realized he needed his blanket, which I had forgotten at a faraway hotel.",
            "I took a drive to get my baby to sleep.",
            "I took my baby for a drive and she fell asleep in the car."
        ],
        "obs1": "My baby would not go to sleep last night.",
        "obs2": "I wound up driving for hours."
    },
'''
print(len(sent_lst))

if MODE == 'ar':
    model_name = 'predictability/diff_models/roc_e=20_b=32_m=gpt2_wikitext-103-raw-v1_101_wp_pad_infill'
    model_name = 'predictability/diff_models/roc_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_pad_infill_v2'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,  # path to the AR model trained for LMing this task.
    ).cuda()
    tokenizer2 = load_tokenizer('roc', 'random',
                               'predictability/diffusion_models_v7/diff_roc_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart')
    vocab = {v: k for k, v in tokenizer2.items()}
    print(len(tokenizer2), len(vocab), 'loaded vocabs')

    outfile='ar_sample_full_test_v2.json'
    filehandle = open(outfile, 'w')

for idx, (key, val) in enumerate(sent_lst.items()):
    # if idx <= 499:
    #     continue
    # if idx >= 500:
    #     continue
    # if idx != 684:
    #     continue

    if MODE == 'diff':
        partial_seq = f"{val['obs1']} " + "PAD "*10 + f"{val['obs2']}"
        word_lst = [x.text for x in tokenizer(partial_seq)]
        partial_seq = " ".join(word_lst)
        print(partial_seq, idx)
        # partial_seq = "Brenna and I used to be best friends . PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD We never talked again ."
        COMMAND = "python ../scripts/infill.py " \
                  "--model_path predictability/diffusion_models_v7/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long/ema_0.9999_800000.pt " \
                  " --batch_size 50  " \
                  f"--partial_seq \'{partial_seq}\' " \
                  f"--eval_task_ infill --notes {SPLIT}_{idx} " \
                  f"--out_dir ../anlg_results"
        os.system(COMMAND)
        torch.cuda.empty_cache()
    elif MODE == 'ar':
        partial_seq = f"{val['obs1']} " + f"{val['obs2']}"
        print(partial_seq)
        word_idx_lst = [vocab['START']] + [vocab.get(x.text, vocab['UNK']) for x in tokenizer(partial_seq)]
        init_prompt = torch.LongTensor(word_idx_lst).cuda().unsqueeze(0)
        print(init_prompt.shape)
        # sample_out = model.generate(init_prompt, do_sample=True, max_length=64, top_k=len(vocab))
        if 'sample' in outfile:
            print('sampling 50 examples.')
            init_prompt = init_prompt.expand(50, -1)
            sample_out = model.generate(init_prompt, do_sample=True, max_length=64, top_k=len(vocab))
        else:
            sample_out = model.generate(init_prompt, do_sample=False, num_beam=4, max_length=64, top_k=len(vocab))

        print(sample_out.shape)
        sample_out = sample_out[:, init_prompt.size(1):]
        # decode
        if 'sample' in outfile:
            sample_lst = []
            for examp in sample_out:
                sample = examp.tolist()
                words_sample = [tokenizer2[s] for s in sample]
                tempsent = [x for x in words_sample if x != 'PAD']
                if tempsent[0] == 'START':
                    tempsent = tempsent[1:]
                if tempsent[-1] == 'END':
                    tempsent = tempsent[:-1]
                result_sent = " ".join(tempsent)
                sample_lst.append(result_sent)
            out_dict = {'idx': idx,
                        'obs1': val['obs1'],
                        'obs2': val['obs2'],
                        'samples': sample_lst}
            print(json.dumps(out_dict), file=filehandle)
        else:
            sample = sample_out[0].tolist()
            words_sample = [tokenizer2[s] for s in sample]
            tempsent = [x for x in words_sample if x != 'PAD']
            if tempsent[0] == 'START':
                tempsent = tempsent[1:]
            if tempsent[-1] == 'END':
                tempsent = tempsent[:-1]
            result_sent = " ".join(tempsent)
            out_dict = {'idx':idx,
                        'obs1':val['obs1'],
                        'obs2':val['obs2'],
                        'sample':result_sent}
            print(json.dumps(out_dict), file=filehandle)
filehandle.close()
print(f'written to {outfile}')






