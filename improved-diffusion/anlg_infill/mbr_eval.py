import os, sys, json
import glob
from functools import partial
sys.path.insert(0, 'e2e-metrics')
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from metrics.pymteval import BLEUScore, NISTScore
from nltk.translate.meteor_score import meteor_score
from parse import *
import json
import sys, os, torch
from spacy.lang.en import English
import ast
from transformers import BertForMaskedLM, BertTokenizer

MODE = sys.argv[1] # ar or diff
SPLIT = sys.argv[2] # val or test
OUT_PATH = sys.argv[3] # output path.
INPUT_PATH = sys.argv[4] # input path. e.g. diffusion_lm/improved-diffusion/anlg_results/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long.ema_0.9999_800000.pt.infill_infill

def load_results_simple(path):
    with open(path, 'r') as f:
        full_result_dict = json.load(f)
    return full_result_dict

def post_process(filename, fileout, tokenizer_spacy):
    # filename = 'diffusion_lm/improved-diffusion/anlg_results/diff_roc_mbr.json2'
    bert_model = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForMaskedLM.from_pretrained(bert_model).cuda()
    fileout_handle = open(fileout, 'w')

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
            model_inputs = tokenizer(sent, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
            model_out = model(**model_inputs)
            mask_words = model_inputs['input_ids'] == tokenizer.mask_token_id
            masked_logits = model_out.logits[mask_words].view(-1, model_out.logits.size(-1))
            # take argmax from this.
            max_cands = torch.max(masked_logits, dim=-1)
            indices = max_cands.indices
            model_inputs['input_ids'][mask_words] = indices
            out = tokenizer.batch_decode(model_inputs['input_ids'].tolist(),
                                         skip_special_tokens=True)[0]
            print(out)
            word_lstout = [x.text for x in tokenizer_spacy(out)]
            word_lst1 = [x.text for x in tokenizer_spacy(example['obs1'])]
            word_lst2 = [x.text for x in tokenizer_spacy(example['obs2'])]
            example['sample'] = " ".join(word_lstout[len(word_lst1):-len(word_lst2)])
            print(example['sample'])
            print()


        else:
            print('NO NEED THIS FIX. ')


        print(json.dumps(example), file=fileout_handle)

    fileout_handle.close()



def load_results(sent_lst, tokenizer):
    # target_file = f"{INPUT_PATH}_*.json"
    # target_file = glob.glob(target_file)
    # print([x for x in target_file if 'val' not in x and 'test' not in x])
    # 10/0
    full_result_dict = {}
    failed_instances = []
    found_idx = []
    sent_lst_lst = list(sent_lst.items())
    for idx, (key, val) in enumerate(sent_lst_lst):
        # if idx < 2500: continue
        if idx in full_result_dict.keys(): continue
        word_lst1 = [x.text for x in tokenizer(val['obs1'])]
        word_lst2 = [x.text for x in tokenizer(val['obs2'])]
        # target_file = f"diffusion_lm/improved-diffusion/anlg_results/diff_roc_pad_rand128_" \
        #               f"transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long.ema" \
        #               f"_0.9999_800000.pt.infill_infill_*_{SPLIT}_{idx}.json"
        target_file = f"{INPUT_PATH}_*_{SPLIT}_{idx}.json"

        file_lst = glob.glob(target_file)
        # print(file_lst, target_file)
        try:
            assert len(file_lst) == 1
        except:
            print('the file must have existed in a batched version')
            # if SPLIT == 'val': assert False
            # if idx % 100 == 1: idx = idx-1
            target_file = f"{INPUT_PATH}_*_{idx}.json"
            file_lst = glob.glob(target_file)
            print(file_lst, target_file)
            print(file_lst)
        target_file = file_lst[0]
        if "x128" in target_file:
            infill_lst = []
            with open(target_file, 'r') as f:
                for line in f:
                    example = json.loads(line)[0]
                    infill_ = example.split()[len(word_lst1):-len(word_lst2)]
                    # print(len(infill_))
                    # print(infill_, example)
                    # assert len(infill_) == 10
                    infill_=' '.join(infill_)
                    # print(infill_)
                    infill_lst.append(infill_)
            result_dict = {
                "pred_samples": infill_lst,
                "sample": None,
                "obs1": val['obs1'],
                "obs2": val['obs2']
            }
            full_result_dict[idx] = result_dict
        else:
            with open(target_file, 'r') as f:
                for line in f:
                    example = ast.literal_eval(line.strip())
                    index, template = list(example.keys())[0]
                    print(index, idx)
                    if int(index) < int(idx):
                        continue
                    assert int(index) == int(idx)
                    found_idx.append(idx)
                    example = list(example.values())[0]
                    kk, val = sent_lst_lst[idx]
                    word_lst1 = [x.text for x in tokenizer(val['obs1'])]
                    word_lst2 = [x.text for x in tokenizer(val['obs2'])]
                    infill_lst = [" ".join(xx.split()[len(word_lst1):-len(word_lst2)]) for xx in example]
                    result_dict = {
                        "pred_samples": infill_lst,
                        "sample": None,
                        "obs1": val['obs1'],
                        "obs2": val['obs2']
                    }
                    full_result_dict[idx] = result_dict
                    idx += 1

    with open('full_diff_test_outputs_aug.json', 'w') as f:
        json.dump(full_result_dict, f)
    return full_result_dict


# read files.
def mbr(result_lst, total_len, sample_size, utility):
    result = []
    for i in range(total_len):
        example_set = result_lst[i * sample_size:(i + 1) * sample_size]
        # print(example_set)
        score_dict = {}
        for idx in range(len(example_set)):
            y = example_set[idx]
            utility_lst = []
            for idx_x in range(len(example_set)):
                if idx_x != idx:
                    utility_lst.append(utility(example_set[idx_x], y))
            score_dict[idx] = np.array(utility_lst).mean()
        # print(score_dict)
        best_y = sorted(score_dict.items(), key=lambda item: item[1])[-1]
        result.append(example_set[best_y[0]])
        # print(best_y)

    return result


def bleu_score(scorer, sent_sys, sents_ref):
    scorer.reset()
    scorer.append(sent_sys, [sents_ref])
    return scorer.score()


def meteor_score2(pred, ref):
    meteor = meteor_score([ref.split()], pred.split())
    return meteor

def apply_mbr_func(full_result_dict, outpath, sent_lst):
    assert len(sent_lst) == len(full_result_dict)
    out_handle = open(outpath, 'w')
    count = 0
    for idx, val in full_result_dict.items():
        infill_lst = val['pred_samples']
        print(count, idx )
        assert count == int(idx)
        count += 1
        sample_size = len(infill_lst)
        total_len = 1
        mteval_scorers = [BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()]
        result_lst = mbr(infill_lst, total_len, sample_size, partial(bleu_score, mteval_scorers[1]))
        print(infill_lst)
        print(result_lst)
        result_str = result_lst[0]
        result_dict = {
            "pred_samples": infill_lst,
            "sample": result_str,
            "obs1": val['obs1'],
            "obs2": val['obs2']
        }
        print(json.dumps(result_dict), file=out_handle)
    out_handle.close()
    print(f'written to {outpath}')
    return

if SPLIT == 'val':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/dev_cleanup.json'
elif SPLIT == 'test':
    source_file = 'diffusion_lm/ROCstory/anlg/anlg/test_cleanup_no_label.json'
else:
    assert False, "invalid split"

with open(source_file, 'r') as f:
    sent_lst = json.load(f)



if MODE == 'diff':
    nlp = English()
    tokenizer = nlp.tokenizer
    # load_results(sent_lst, tokenizer)
    # 10/0
    decoded_dict = load_results_simple(INPUT_PATH)
    ############3
    # small_decoded_dict = {}
    # for i in range(10):
    #     small_decoded_dict[i] = decoded_dict[str(i)]
    # decoded_dict = small_decoded_dict
    # small_sent_lst = {}
    # for k, v in sent_lst.items():
    #     if len(small_sent_lst) > 9: break
    #     small_sent_lst[k] = v
    # sent_lst = small_sent_lst
    ############3
    outpath = OUT_PATH
    apply_mbr_func(decoded_dict, outpath, sent_lst)
    post_process(outpath, outpath+'.clean.json', tokenizer)

    #
    # # load_results(sent_lst, tokenizer)
    # # 10/0
    # print(len(sent_lst))
    # for idx, (key, val) in enumerate(sent_lst.items()):
    #     # if idx < 518: continue
    #     if idx > 900:
    #         break
    #     # change the matching method.
    #     word_lst1 = [x.text for x in tokenizer(val['obs1'])]
    #     word_lst2 = [x.text for x in tokenizer(val['obs2'])]
    #     # partial_seq = f"{val['obs1']} " + "PAD " + f"{val['obs2']}"
    #     # word_lst = [x.text for x in tokenizer(partial_seq)]
    #     # partial_seq = " ".join(word_lst)
    #     # partial_seq = partial_seq.replace('PAD', '{}')
    #     # print(partial_seq, idx)
    #
    #     # target_file = f"diffusion_lm/improved-diffusion/anlg_results/diff_roc_pad_rand128_" \
    #     #               f"transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long.ema" \
    #     #               f"_0.9999_800000.pt.infill_infill_*_{SPLIT}_{idx}.json"
    #     target_file = f"{INPUT_PATH}_*_{SPLIT}_{idx}.json"
    #
    #     file_lst = glob.glob(target_file)
    #     print(file_lst, target_file)
    #     assert len(file_lst) == 1
    #     target_file = file_lst[0]
    #     # print(target_file)
    #     infill_lst = []
    #     with open(target_file, 'r') as f:
    #         for line in f:
    #             example = json.loads(line)[0]
    #             # print(example, partial_seq)
    #             # infill_ = parse(partial_seq, example)
    #             # print(example)
    #             infill_ = example.split()[len(word_lst1):-len(word_lst2)]
    #             # print(len(infill_))
    #             # print(infill_, example)
    #             # assert len(infill_) == 10
    #             infill_=' '.join(infill_)
    #             # print(infill_)
    #             infill_lst.append(infill_)
    #     infill_lst = infill_lst
    #     sample_size = len(infill_lst)
    #     total_len = 1
    #     mteval_scorers = [BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()]
    #     result_lst = mbr(infill_lst, total_len, sample_size, partial(bleu_score, mteval_scorers[1]))
    #     print(infill_lst)
    #     print(result_lst)
    #     result_str = result_lst[0]
    #     result_dict = {
    #         "pred_samples": infill_lst,
    #         "sample":result_str,
    #         "obs1": val['obs1'],
    #         "obs2": val['obs2']
    #     }
    #     print(json.dumps(result_dict), file=out_handle)
    #
    # out_handle.close()
    # print(f'written to {outpath}')

elif MODE == 'ar':
    outpath = OUT_PATH #'diffusion_lm/improved-diffusion/anlg_results/ar_full_mbr.json'
    out_handle = open(outpath, 'w')
    sample_file = INPUT_PATH #'diffusion_lm/improved-diffusion/anlg_results/ar_sample_500_v2.json'
    nlp = English()
    tokenizer = nlp.tokenizer
    print(len(sent_lst))
    sample_lst = []
    with open(sample_file, 'r') as f:
        for line in f:
            sample_dict = json.loads(line)
            sample_lst.append(sample_dict)

    for idx, (key, val) in enumerate(sent_lst.items()):
        # if idx < 109: continue
        # if idx > 499:
        #     break
        infill_lst = sample_lst[idx]['samples']
        sample_size = len(infill_lst)
        total_len = 1
        mteval_scorers = [BLEUScore(), BLEUScore(smoothing=1.0), NISTScore()]
        result_lst = mbr(infill_lst, total_len, sample_size, partial(bleu_score, mteval_scorers[1]))
        print(infill_lst)
        print(result_lst)
        result_str = result_lst[0]
        result_dict = {
            "pred_samples": infill_lst,
            "sample": result_str,
            "obs1": val['obs1'],
            "obs2": val['obs2']
        }
        print(json.dumps(result_dict), file=out_handle)

    out_handle.close()
    print(f'written to {outpath}')

    post_process(outpath, outpath + '.clean.json', tokenizer)

# print(file+'.clean')
# with open(file+'.clean', 'w') as f:
#     for line in result_lst:
#         print(line, file=f)

