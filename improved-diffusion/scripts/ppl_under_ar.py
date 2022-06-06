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

from improved_diffusion.test_util import  load_results

def main():
    args = create_argparser().parse_args()
    # set_seed(101)
    set_seed(108)


    if args.mode == 'eval':

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, # path to the AR model trained for LMing this task.
        ).cuda()

        model.eval()

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
        reverse_tokenizer = {v:k for k,v in tokenizer.items()}

        agg_loss = []
        for x in text_samples:
            # print(x)
            tokenized_x = [reverse_tokenizer[s] for s in x]
            # print(tokenized_x)
            tokenized_x = torch.LongTensor(tokenized_x).cuda()
            labels = tokenized_x.clone()
            labels[labels==reverse_tokenizer['PAD']] = -100
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
    elif args.mode == 'gen':
        generate(args)

    elif args.mode == 'gen_gpt2':
        generate_gpt2(args)

def generate_gpt2(args):

    print(f'loading from {args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # path to the AR model trained for LMing this task.
    ).cuda()

    # load tokenizer.
    sample_out_lst = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    for i in range(1000):
        sample_out = model.generate(do_sample=True,  max_length=64,
                                    top_k=len(tokenizer), top_p=args.top_p,
                                    num_return_sequences=512,
                                    pad_token_id=tokenizer.eos_token_id)
        print(sample_out.shape)
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