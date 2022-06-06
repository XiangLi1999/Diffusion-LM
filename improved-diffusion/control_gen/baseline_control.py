# syntax, semantics, etc...
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from nltk.tree import Tree

from improved_diffusion.test_util import  load_results



def remove_leaves(tree_):
    # simple_increm = 0
    for s in tree_.subtrees(lambda t: t.height() == 2):
        s[0] = '*'
        s._label = ''
    return tree_

def main():
    args = create_argparser().parse_args()
    set_seed(108)

    # toy1 = 'START Alimentum is not a family - friendly place , located in city centre . \n END'.split()
    # toy1 = 'START Located in riverside area , Alimentum restaurant is a place to bring the whole family . \n END'.split()
    toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating',
            '.', 'Prices', 'start', 'at', '£', '30', '.', 'END']

    if args.mode == 'tree':

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, # path to the AR model trained for LMing this task.
        ).cuda()
        model.eval()

        if args.finetune == 'yes':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:

            pass

        control_label_lst = []
        with open('diffusion_lm/improved-diffusion/control_gen/target_tree.json', 'r') as controlf:
            for line in controlf:
                control_label_lst.append(json.loads(line))

        result_dict = {}
        for label_class_dict in control_label_lst:  # control_label_lst[:100]:
            '''
                input_strings = [" ".join(pos_) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                                         for (pos_, seq) in zip(pos_lst, examples['text'])]
            '''
            parse_tree = Tree.fromstring(label_class_dict['tree'])
            print(parse_tree)
            parse_tree = remove_leaves(parse_tree)

            prompt_strings =  parse_tree._pformat_flat("", "()", False) + tokenizer.bos_token
            prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
            out_text = generate_samples(args, prompt_ids['input_ids'].cuda(), model, tokenizer)
            result_dict[(label_class_dict['tree'],)] = out_text
            print(len(out_text))

        fout = open(args.output_text, 'w')
        for k, word_lst in result_dict.items():
            print({k: word_lst}, file=fout)
        fout.close()

        # # load trees.
        # import benepar
        # parser = benepar.Parser("benepar_en3")
        # input_sentence1 = benepar.InputSentence(
        #     words=toy1[1:-1],
        # )
        # parse_lst = list(parser.parse_sents([input_sentence1]))[0]
        # print(parse_lst)
        # parse_lst = remove_leaves(parse_lst)
        # prompt_strings = parse_lst._pformat_flat("", "()", False) + tokenizer.bos_token
        # print(prompt_strings)
        # prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
        # print(prompt_ids['input_ids'].shape)
        #
        # generate_gpt2(args, prompt_ids['input_ids'].cuda())

        # eval(args)
    if args.mode == 'spans':

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, # path to the AR model trained for LMing this task.
        ).cuda()
        model.eval()

        if args.finetune == 'yes':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            import benepar
            parser = benepar.Parser("benepar_en3")
            tree_vocab = parser._parser.config["label_vocab"]

            model_path = 'predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart'
            tokenizer2 = load_tokenizer('e2e-tgt', 'random', model_path)
            tokenizer = {v: k for k, v in tokenizer2.items()}
            print(len(tokenizer), len(tokenizer2), 'loaded vocabs')

            print('update the vocab to include tree vocabs')
            print(len(tokenizer))
            for x in tree_vocab.keys():
                tokenizer[x] = len(tokenizer)
            print('update the vocab to include indices')
            # tokenizer.add_tokens([str(xx) for xx in range(64)])
            for x in range(64):
                if str(x) not in tokenizer:
                    tokenizer[str(x)] = len(tokenizer)
            vocab_dict = tokenizer
            rev_tokenizer = {v: k for k, v in vocab_dict.items()}
        print(len(tokenizer))


        control_label_lst = []
        with open('diffusion_lm/improved-diffusion/control_gen/target_spans.json', 'r') as controlf:
            for line in controlf:
                control_label_lst.append(json.loads(line))

        result_dict = {}
        for span_info in control_label_lst:  # control_label_lst[:100]:
            (a,b,c) = span_info['spans'][0]
            if args.finetune == 'yes':
                prompt_strings = f"{a}, {b}, {c}" + tokenizer.bos_token
                print(prompt_strings)
                prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
                out_text = generate_samples(args, prompt_ids['input_ids'].cuda(), model, tokenizer)
            else:
                prompt_ids = [vocab_dict.get(x, vocab_dict['UNK']) for x in f"{a} {b} {c}".split()] + [0]
                print(prompt_ids)
                prompt_ids = torch.LongTensor(prompt_ids).unsqueeze(0)
                out_text = generate_samples_from_scratch(args, prompt_ids.cuda(), model, tokenizer, rev_tokenizer)
            # str(label_class_dict['spans'][0]),
            result_dict[str(span_info['spans'][0])] = out_text
            print(len(out_text))

        fout = open(args.output_text, 'w')
        for k, word_lst in result_dict.items():
            print({(k,): word_lst}, file=fout)
        fout.close()
    elif args.mode == 'pos':
        import spacy_stanza
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,  # path to the AR model trained for LMing this task.
        ).cuda()
        model.eval()

        if args.finetune == 'yes':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            pass

        control_label_lst = []
        with open('diffusion_lm/improved-diffusion/control_gen/target_pos.json', 'r') as controlf:
            for line in controlf:
                control_label_lst.append(json.loads(line))
        print(control_label_lst[:5])

        result_dict = {}
        for label_class_dict in control_label_lst:  # control_label_lst[:100]:
            '''
                input_strings = [" ".join(pos_) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                                         for (pos_, seq) in zip(pos_lst, examples['text'])]
            '''
            gold_pos = label_class_dict['pos'][1:-1] # remove START, END.
            words_ = label_class_dict['words_']
            print(gold_pos, 'target POS tagging sequences', tokenizer.bos_token)
            prompt_strings = " ".join(gold_pos) + tokenizer.bos_token
            prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
            out_text = generate_samples(args, prompt_ids['input_ids'].cuda(), model, tokenizer )
            result_dict[tuple(gold_pos)] = out_text
            print(len(out_text))

        fout = open(args.output_text, 'w')
        for k, word_lst in result_dict.items():
            print({k:word_lst}, file=fout)
        fout.close()


        # tagger = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
        # toy1 = 'START The Mill is a coffee shop with an expensive menu near The Sorrento . \n END'.split()
        # toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.',
        #         'Prices', 'start', 'at', '£', '30', '.', '\n', 'END']
        # sent_full = " ".join(toy1[1:-1])
        # doc = tagger(sent_full)
        # gold_pos = [token.pos_ for token in doc]
        # print(gold_pos, 'target POS tagging sequences')
        # prompt_strings = " ".join(gold_pos) + tokenizer.bos_token
        # prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
        # generate_gpt2(args, prompt_ids['input_ids'].cuda())

    elif args.mode == 'attribute':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,  # path to the AR model trained for LMing this task.
        ).cuda()
        model.eval()

        if args.finetune == 'yes':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            pass

        control_label_lst = []
        with open('diffusion_lm/improved-diffusion/control_gen/target_attribute.json', 'r') as controlf:
            for line in controlf:
                control_label_lst.append(json.loads(line))
        print(control_label_lst[:5])

        result_dict = {}
        for label_class in control_label_lst:  # control_label_lst[:100]:
            prompt_strings = " ".join(label_class) + tokenizer.bos_token
            '''
            input_strings = [
                        " ".join(attributes) + tokenizer.bos_token + " ".join(words) + tokenizer.eos_token
                        for (words, attributes) in examples['text']]
            '''
            print(label_class, 'target attribute sequences', tokenizer.bos_token)
            prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
            out_text = generate_samples(args, prompt_ids['input_ids'].cuda(), model, tokenizer)
            result_dict[tuple(label_class)] = out_text
            print(len(out_text))

        fout = open(args.output_text, 'w')
        for k, word_lst in result_dict.items():
            print({k: word_lst}, file=fout)
        fout.close()

    elif args.mode == 'control_len':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,  # path to the AR model trained for LMing this task.
        ).cuda()
        model.eval()

        if args.finetune == 'yes':
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            pass


        result_dict = {}
        for label_class in range(10, 41):  # control_label_lst[:100]:
            tgt_len = label_class-2
            prompt_strings = f"{tgt_len}" + tokenizer.bos_token
            print(label_class, 'target attribute sequences', tokenizer.bos_token)
            prompt_ids = tokenizer([prompt_strings], return_tensors='pt')
            out_text = generate_samples(args, prompt_ids['input_ids'].cuda(), model, tokenizer)
            result_dict[tuple([label_class])] = out_text
            print(len(out_text))

        fout = open(args.output_text, 'w')
        for k, word_lst in result_dict.items():
            print({k: word_lst}, file=fout)
        fout.close()

        # generate_gpt2(args)


def eval(args):
    text_samples = []
    if args.input_text.endswith('json'):
        with open(args.input_text, 'r') as f:
            for line in f:
                text_samples.append(json.loads(line)[0].split(' '))
    else:
        with open(args.input_text, 'r') as f:
            for line in f:
                text_samples.append(line.strip().split())

    # tokenize
    # load tokenizer.
    tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
    # print(args.modality, tokenizer, args.experiment)
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}

    agg_loss = []
    for x in text_samples:
        # print(x)
        tokenized_x = [reverse_tokenizer[s] for s in x]
        # print(tokenized_x)
        tokenized_x = torch.LongTensor(tokenized_x).cuda()
        labels = tokenized_x.clone()
        labels[labels == reverse_tokenizer['PAD']] = -100
        model_output = model(tokenized_x, labels=labels)
        # print(model_output.loss)
        agg_loss.append(model_output.loss.item())

    print(f'\nthe mean loss is {torch.tensor(agg_loss).mean()} for {args.input_text}', )
    print('-' * 50)
    if 'infill' in args.input_text:
        json_path = os.path.join(os.path.split(args.model_path)[0], 'infill_score_decode.json')
    elif 'ema' in args.model_path:
        json_path = os.path.join(os.path.split(args.model_path)[0], 'ema_score_decode.json')
    else:
        json_path = os.path.join(os.path.split(args.model_path)[0], 'score_decode.json')
    print(f'written to {json_path}')
    json_dict = {
        'score_decode': torch.tensor(agg_loss).mean().item(),
        'source_decode': args.input_text,
    }
    load_results(json_path, json_dict)

def generate_samples(args, prompt, model, tokenizer):
    if args.generation_mode == 'search':
        sample_out = model.generate(prompt, do_sample=False, max_length=200, min_length=prompt.size(1) + 1, num_beams=4,
                                    top_k=len(tokenizer), top_p=args.top_p, num_return_sequences=1,
                                    pad_token_id=tokenizer.pad_token_id)
    else:
        sample_out = model.generate(prompt, do_sample=True, max_length=200, min_length=prompt.size(1)+1,
                                    top_k=len(tokenizer), top_p=args.top_p, num_return_sequences=1,
                                    pad_token_id=tokenizer.pad_token_id)
    sample_out_lst = sample_out[:, prompt.size(1):]
    # sample_out_lst.append(sample_out.cpu())
    # sample_out_lst = torch.cat(sample_out_lst, dim=0)
    text_out = []
    for sample in sample_out_lst:
        sample = sample.tolist()
        words_sample = tokenizer.decode(sample, skip_special_tokens=True)
        text_out.append(words_sample)
    return text_out

def generate_samples_from_scratch(args, prompt, model, tokenizer, rev_tokenizer):
    print('generating from scratch')
    if args.generation_mode == 'search':
        sample_out = model.generate(prompt, do_sample=False, max_length=200, min_length=prompt.size(1) + 1, num_beams=4,
                                    top_k=len(tokenizer), top_p=args.top_p, num_return_sequences=1,
                                    pad_token_id=tokenizer['PAD'], eos_token_id=tokenizer['END'])
    else:
        sample_out = model.generate(prompt, do_sample=True, max_length=200, min_length=prompt.size(1) + 1,
                                    top_k=len(tokenizer), top_p=args.top_p, num_return_sequences=50,
                                    pad_token_id=tokenizer['PAD'], eos_token_id=tokenizer['END'])
    sample_out_lst = sample_out[:, prompt.size(1):]
    # sample_out_lst.append(sample_out.cpu())
    # sample_out_lst = torch.cat(sample_out_lst, dim=0)
    text_out = []
    for sample in sample_out_lst:
        sample = sample.tolist()
        words_sample = " ".join([rev_tokenizer[x] for x in sample])
        text_out.append(words_sample)
    return text_out

def generate_gpt2(args, prompt=None):

    print(f'loading from {args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # path to the AR model trained for LMing this task.
    ).cuda()

    # load tokenizer.
    sample_out_lst = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    sample_out = model.generate(prompt, do_sample=True, max_length=200,
                                top_k=len(tokenizer), top_p=args.top_p, num_return_sequences=50, pad_token_id=tokenizer.pad_token_id)
    sample_out = sample_out[:, prompt.size(1):]
    sample_out_lst.append(sample_out.cpu())
    sample_out_lst = torch.cat(sample_out_lst, dim=0)


    if args.output_text.endswith('json'):
        with open(args.output_text, 'w') as f:
            for sample in sample_out_lst:
                sample = sample.tolist()
                words_sample = tokenizer.decode(sample, skip_special_tokens=True)
                print(json.dumps([words_sample]), file=f)
    else:
        with open(args.output_text, 'w') as f:
            for sample in sample_out_lst:
                sample = sample.tolist()
                words_sample = tokenizer.decode(sample,  skip_special_tokens=True)
                print(words_sample, file=f)

    agg_loss = []
    for tokenized_x in sample_out:
        labels = tokenized_x.clone()
        labels[labels == tokenizer.eos_token_id] = -100
        model_output = model(tokenized_x, labels=labels)
        agg_loss.append(model_output.loss.item())

    print(f'\nthe mean loss is {torch.tensor(agg_loss).mean()}',)
    print('-'*50)

def generate(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # path to the AR model trained for LMing this task.
    ).cuda()

    print(model.transformer.wte)
    # print(model)
    # load tokenizer.
    tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}
    print(len(tokenizer))

    init_prompt = torch.LongTensor([reverse_tokenizer['START']]).view(1,1).expand(50, -1).to(model.device)
    sample_out = model.generate(init_prompt, do_sample=True,  max_length=64,
                                top_k=len(tokenizer), top_p=args.top_p)
    print(sample_out.shape)

    if args.output_text.endswith('json'):
        with open(args.output_text, 'w') as f:
            for sample in sample_out:
                sample = sample.tolist()
                words_sample = [tokenizer[s] for s in sample]
                print(json.dumps([" ".join(words_sample)]), file=f)
    else:
        with open(args.output_text, 'w') as f:
            for sample in sample_out:
                sample = sample.tolist()
                words_sample = [tokenizer[s] for s in sample]
                print(" ".join(words_sample), file=f)

    agg_loss = []
    for tokenized_x in sample_out:
        model_output = model(tokenized_x, labels=tokenized_x)
        agg_loss.append(model_output.loss.item())

    print(f'\nthe mean loss is {torch.tensor(agg_loss).mean()}',)
    print('-'*50)

    ##################

    text_samples = []
    if args.output_text.endswith('json'):
        with open(args.output_text, 'r') as f:
            for line in f:
                text_samples.append(json.loads(line)[0].split(' '))
    else:
        with open(args.output_text, 'r') as f:
            for line in f:
                text_samples.append(line.strip().split())


    agg_loss = []
    for idx, x in enumerate(text_samples):
        # print(x)
        tokenized_x = [reverse_tokenizer[s] for s in x]
        tokenized_x = torch.LongTensor(tokenized_x).cuda()
        # print(tokenized_x)
        # print(sample_out[idx])
        # print((tokenized_x == sample_out[idx]).all())
        model_output = model(tokenized_x, labels=tokenized_x)
        # print(model_output.loss)
        agg_loss.append(model_output.loss.item())

    print(f'\nthe mean loss is {torch.tensor(agg_loss).mean()} for {args.input_text}', )



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50,#10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        finetune='yes',
        generation_mode='sample',
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         input_text='',
                         mode='eval',
                         output_text='',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1, top_p=1.0,)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser





if __name__ == '__main__':
    with torch.no_grad():
        main()