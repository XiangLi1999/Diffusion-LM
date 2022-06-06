# from PIL import Image
# import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, PreTrainedTokenizerFast, \
    PreTrainedTokenizer
from datasets import load_dataset
import sys, os
import torch
# sys.path.insert(0, os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise
from collections import Counter, defaultdict
from functools import partial
from itertools import chain


def load_data_text(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, data_args=None, 
        task_mode='roc', model=None, padding_mode='block', split='train', load_vocab=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    print('hello loading text data. ')

    if data_args.experiment.startswith('random') and model is None:
        model = None
    elif data_args.experiment.startswith('random') and model is not None:
        print('loading initialized random embeddings. ')

    if task_mode == 'roc' or task_mode == 'roc-aug' :
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)
    elif task_mode == 'simple-wiki':
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)

    elif task_mode == 'e2e-tgt':
        print('hello loading e2e-tgt. ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)
    elif task_mode == 'yelp':
        print('hello loading yelp ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)

    elif task_mode == 'commonGen' or task_mode == 'commonGen-aug':
        print('hello loading common-gen ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)

    elif task_mode == 'e2e':
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)

    elif task_mode == 'book':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        training_data, model = get_corpus_book(data_args, tokenizer, model, image_size,
                                              padding_mode=padding_mode, split=split,)

    if data_args.modality in ['roc-aug', 'roc', 'book', 'yelp', 'commonGen', 'commonGen-aug'] and data_args.cache_mode=='no':
        dataset = TextDataset_NoCache(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
            model_emb=model
        )
    else:
        dataset = TextDataset(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
        )

    if deterministic:

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=False,
            num_workers=1,
        )

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=True,
            num_workers=1,
        )
    while True:
        yield from data_loader

def helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, seqlen, data_args):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for (src_ids, input_ids) in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            tokenized_src = [vocab_dict.get(x, vocab_dict['UNK']) for x in src_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
            group_lst['src_ids'].append(tokenized_src)

        print(group_lst['word_ids'][:2])
        print('padding mode is pad')
        max_length = seqlen
        group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        print(max_src_length, seqlen)
        max_src_length = min(seqlen, max_src_length)
        group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                            vocab_dict['PAD'],
                                                                            max_src_length,
                                                                            return_mask=True)


        for input_ids, src_ids, src_mask in zip(group_lst['word_ids'], group_lst['src_ids'],
                                      group_lst['src_mask']):
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            result_train_lst.append({'input_ids': input_ids,
                                     'hidden_states': hidden_state.cpu().tolist(),
                                     'src_ids':src_ids,
                                     'src_mask':src_mask
                                     })

    return result_train_lst

def helper_tokenize_stream(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    import psutil
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    from datasets import Dataset as Dataset2
    raw_datasets = Dataset2.from_dict({'text':sentence_lst})
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


    def tokenize_function(examples):
        if isinstance(vocab_dict, dict):
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
        elif isinstance(vocab_dict, PreTrainedTokenizerFast):
            examples['text'] = [" ".join(seq) for seq in examples['text']]
            input_ids = vocab_dict(examples['text'], add_special_tokens=True)['input_ids']
        result_dict = {'input_ids': input_ids}
        # clm input could be much much longer than block_size
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    if padding_mode == 'block':
        block_size = seqlen
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        def pad_function(group_lst):
            max_length = seqlen
            if isinstance(vocab_dict, dict):
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
            else:
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
            return group_lst

        # Process.memory_info is expressed in bytes, so convert to megabytes
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        lm_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=1,
            desc=f"padding",
        )


    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    import datasets
    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets

def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for input_ids in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
        print(group_lst['word_ids'][:2])

        if padding_mode == 'block':
            print('padding mode is block')
            concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
            total_length = len(concatenated_examples[list(group_lst.keys())[0]])
            block_size = seqlen
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            group_lst = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        elif padding_mode == 'pad':
            print('padding mode is pad')
            max_length = seqlen
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)

        for input_ids in group_lst['word_ids']:
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            elif data_args.experiment == 'glove':
                hidden_state = model(torch.tensor(input_ids))
            result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})

    return result_train_lst

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = torch.tensor(np.array(split_line[1:], dtype=np.float64))
            # embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model

def load_glove(vocab):
    model = torch.nn.Embedding(len(vocab), 50)
    glove_model = load_glove_model('predictability/glove/glove.6B.50d.txt')
    array_lst = []
    count_ = 0
    for word, idx in vocab.items():
        if word in glove_model:
            array_lst.append(glove_model[word])
        else:
            count_ += 1
            array_lst.append(torch.randn(50))
    print(f'{count_} out of {len(vocab)} is initialized. ')
    array_lst = torch.stack(array_lst)
    print(torch.norm(array_lst, dim=-1).mean())
    model.weight.data = array_lst
    return model


def get_corpus_rocstory(data_args, model, image_size, padding_mode='block',
                        split='train', load_vocab=None):
    import csv, torch, json
    from spacy.lang.en import English

    if data_args.experiment_mode == 'lm':
        if data_args.modality == 'roc':
            print('loading dataset from ROCStory')
            nlp = English()
            tokenizer = nlp.tokenizer
            sentence_lst = []
            print(f'loading from {data_args.roc_train}')
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.roc_train}/roc_train.json'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.roc_train}/roc_valid.json'
            else:
                assert False, "invalid split for ROC dataset"

            with open(path, 'r') as roc_reader:
                for row in roc_reader:
                    sentences = json.loads(row)[0].strip()
                    word_lst = [x.text for x in tokenizer(sentences)]
                    sentence_lst.append(word_lst)

            # with open(data_args.roc_train, 'r') as csvfile:
            #     roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            #     for row in roc_reader:
            #         # tokenize.
            #         sentences = " ".join(row[2:])
            #         word_lst = [x.text for x in tokenizer(sentences)]
            #         sentence_lst.append(word_lst)
            # sentence_lst = sentence_lst[1:]
            print(sentence_lst[:2])
        if data_args.modality == 'roc-aug':
            print('loading dataset from ROCStory')
            nlp = English()
            tokenizer = nlp.tokenizer
            sentence_lst = []
            if split == 'train':
                print('loading form the TRAIN set')
                path_lst = [f'{data_args.roc_train}/roc_train.json']
                path_lst.append('diffusion_lm/improved-diffusion/diff_models/rocstories_gptj.txt')
                # path_lst.append('diffusion_lm/improved-diffusion/cache/ar_model_augment_roc.json')
                # path_lst.append('diffusion_lm/improved-diffusion/cache/ar_model_augment_roc2.json')

            elif split == 'valid':
                print('loading form the VALID set')
                path_lst = [f'{data_args.roc_train}/roc_valid.json']
            else:
                assert False, "invalid split for ROC dataset"

            print(path_lst)
            for path in path_lst:
                if path.endswith('txt'):
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = row.strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
                else:
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = json.loads(row)[0].strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2],sentence_lst[-2:], 'dataset size=',len(sentence_lst))
        elif data_args.modality == 'simple-wiki':
            print('loading dataset from simple wikipedia')
            sentence_lst = []
            with open(data_args.wiki_train, 'r') as ff:
                for row in ff:
                    word_lst = row.lower().split()
                    sentence_lst.append(word_lst)
            print(sentence_lst[:2])
        elif data_args.modality == 'e2e-tgt':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/src1_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/src1_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/src1_test.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        word_lst = row.split('||')[1]
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])

        elif data_args.modality == 'yelp':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.yelp_train}/yelpnlg-train.csv'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.yelp_train}/yelpnlg-dev.csv'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.yelp_train}/yelpnlg-test.csv'
            if split in ['train', 'valid', 'test']:

                with open(path, 'r') as csvfile:
                    yelp_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
                    for row in yelp_reader:
                        sentences = row[1]
                        word_lst = [x.text for x in tokenizer(sentences)]
                        sentence_lst.append(word_lst)
                sentence_lst = sentence_lst[1:]
            print(sentence_lst[:2])

        elif data_args.modality == 'commonGen':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.commonGen_train}/commongen.train.jsonl'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.commonGen_train}/commongen.dev.jsonl'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.commonGen_train}/commongen.test.jsonl'
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for line in ff:
                        line = json.loads(line)
                        for sentences in line['scene']:
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2])

        elif data_args.modality == 'commonGen-aug':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.commonGen_train}/commongen.train.jsonl'
                path_lst = [f'{data_args.roc_train}/roc_train.json']
                path_lst.append('diffusion_lm/improved-diffusion/diff_models/rocstories_gptj.txt')
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.commonGen_train}/commongen.dev.jsonl'
                path_lst = []
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.commonGen_train}/commongen.test.jsonl'
                path_lst = []

            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for line in ff:
                        line = json.loads(line)
                        for sentences in line['scene']:
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2])
            import itertools
            for path in path_lst:
                if path.endswith('txt'):
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = row.strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            spl = [[]]
                            for x, y in itertools.groupby(word_lst, lambda z: z == '.'):
                                spl[-1].extend(y)
                                if x: spl.append([])
                            sentence_lst.extend(spl[:-1])
                else:
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = json.loads(row)[0].strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            spl = [[]]
                            for x, y in itertools.groupby(word_lst, lambda z: z == '.'):
                                spl[-1].extend(y)
                                if x: spl.append([])
                            sentence_lst.extend(spl[:-1])

            print(sentence_lst[-2:])


        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for input_ids in sentence_lst:
                counter.update(input_ids)

    if data_args.experiment_mode == 'conditional_gen':
        if data_args.modality == 'e2e':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                path = f'{data_args.e2e_train}/src1_train.txt'
                with open(path, 'r') as ff:
                    for row in ff:
                        src_lst, word_lst = row.split('||')
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        src_lst = [x.text for x in tokenizer(src_lst)]
                        sentence_lst.append((src_lst, word_lst))
            elif split == 'valid':
                path = f'{data_args.e2e_train}/src1_valid.txt'
                sentence_lst = read_e2e_files(path, data_args, tokenizer)
            print(sentence_lst[:2])
        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for (src_ids, input_ids) in sentence_lst:
                counter.update(input_ids)
                counter.update(src_ids)

    if load_vocab is None:
        vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
        for k, v in counter.items():
            if v > 10:
                vocab_dict[k] = len(vocab_dict)
        print(len(counter), len(vocab_dict))

        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        print(f'save the vocab to {path_save_vocab}')
        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        if not os.path.exists(path_save_vocab):
            print(f'save the vocab to {path_save_vocab}')
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(data_args.checkpoint_path)
            else:
                assert False, "invalid type of vocab_dict"



    if model is None and data_args.experiment == 'random':
        model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)
    elif data_args.experiment == 'gpt2_pre_compress':
        assert model is not None
    elif data_args.experiment == 'glove':
        assert data_args.in_channel == 50
        model = load_glove(vocab_dict)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    if not os.path.exists(path_save) and data_args.experiment == 'random':
        torch.save(model.state_dict(), path_save)


    if data_args.experiment_mode == 'lm' and data_args.modality in ['roc-aug', 'roc', 'yelp', 'commonGen', 'commonGen-aug'] \
            and data_args.cache_mode=='no':
        train_dataset = helper_tokenize_stream(sentence_lst, vocab_dict, model, image_size**2, data_args, padding_mode)
        return train_dataset, model
    elif data_args.experiment_mode == 'lm':
        result_train_lst = helper_tokenize_encode(sentence_lst, vocab_dict, model, image_size**2, data_args, padding_mode)
    elif data_args.experiment_mode == 'conditional_gen':
        result_train_lst = helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, image_size ** 2, data_args)
    return {'train': result_train_lst}, model
       

def write_e2e_corr(prompt_lst, file_dict, corr_path):
    print(len(prompt_lst))
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            for line in file_dict[x]:
                print(" ".join(line), file=f)
            print('', file=f)


def write_e2e_src(prompt_lst, corr_path):
    with open(corr_path, 'w') as f:
        for x in prompt_lst:
            print(" ".join(x), file=f)
    return


def read_e2e_files(path, args, tokenizer):
    file_dict = {}
    with open(path, 'r') as f:
        for line in f:
            src_lst, word_lst = line.strip().split('||')
            tgt = tuple([x.text for x in tokenizer(word_lst)])
            src = tuple([x.text for x in tokenizer(src_lst)])
            if src not in file_dict:
                file_dict[src] = []
            file_dict[src].append(tgt)
    temp = '1'
    prompt_text_dict = file_dict
    prompt_text_lst = list(prompt_text_dict.keys())
    gold_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'gold'))
    print("gold dir", gold_dir)
    write_e2e_corr(prompt_text_lst, prompt_text_dict, gold_dir)
    src_dir = os.path.join(args.out_dir, '{}_{}_{}'.format(temp, args.split, 'src'))
    write_e2e_src(prompt_text_lst, src_dir)
    final_lst = [(xx, prompt_text_dict[xx][0]) for xx in prompt_text_lst]
    return final_lst


def get_corpus_book(data_args, tokenizer, model, image_size, padding_mode='block', split='train',):
    max_length = image_size ** 2
    import os
    assert padding_mode == 'block'
    raw_datasets = load_dataset('bookcorpus')
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            'bookcorpus',
            split=f"train[:1%]",
        )
        raw_datasets["train"] = load_dataset(
            'bookcorpus',
            split=f"train[1%:]",
        )
    print(raw_datasets)
    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        output = tokenizer(examples['text'], add_special_tokens=False)
        return output


    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
    )

    print(tokenized_datasets)

    block_size = max_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    print(lm_datasets)

    if model is None:
        if data_args.training_mode.startswith('e2e'):
            print('since its e2e, initialize a dummy embedding' )
            model = torch.nn.Embedding(len(tokenizer), 1)
        else:
            model = torch.nn.Embedding(len(tokenizer), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    if split == 'train':
        return lm_datasets, model
    else:
        lm_datasets['train'] = lm_datasets['validation']
        return lm_datasets, model


class TextDataset(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        if self.model_arch == 'conv-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32).reshape(self.resolution, self.resolution, -1)
            # print(self.eigen_transform.shape)
            if self.eigen_transform  is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)


            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(out_dict.keys())
            return np.transpose(arr, [2, 0, 1]), out_dict
        elif self.model_arch == '1d-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32) # seqlen, dim
            if self.eigen_transform  is not None:
                old_shape = arr.shape
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            arr = np.transpose(arr, [1, 0])
            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(arr.shape)
            return arr, out_dict
        else:
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)
            if self.eigen_transform  is not None:
                old_shape = arr.shape
                # arr = arr.reshape(1, -1) @ self.eigen_transform
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
                
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                # print(arr.dtype)
                # print(self.data_args.noise_level, 'using the noise level.')
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                # print(arr.dtype)

            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            if self.data_args.experiment_mode == 'conditional_gen':
                out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            return arr, out_dict
        # print(arr.dtype)
        # arr = arr.float()
        # print(arr.shape)


class TextDataset_NoCache(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        print(self.resolution)
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            model = self.model_emb
            if self.data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif self.data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor

            if self.model_arch == 'conv-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32).reshape(self.resolution, self.resolution, -1)
                # print(self.eigen_transform.shape)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(out_dict.keys())
                return np.transpose(arr, [2, 0, 1]), out_dict
            elif self.model_arch == '1d-unet':
                arr = np.array(hidden_state,
                               dtype=np.float32)  # seqlen, dim
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)
                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                arr = np.transpose(arr, [1, 0])
                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                # print(arr.shape)
                return arr, out_dict
            else:
                arr = np.array(hidden_state,
                               dtype=np.float32)
                if self.eigen_transform is not None:
                    old_shape = arr.shape
                    # arr = arr.reshape(1, -1) @ self.eigen_transform
                    arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                    arr = arr @ self.eigen_transform['map']
                    arr = arr.reshape(old_shape)

                if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                    # print(arr.dtype)
                    # print(self.data_args.noise_level, 'using the noise level.')
                    arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                    # print(arr.dtype)

                out_dict = {}
                out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
                # out_dict['mapping_func'] = self.mapping_func
                if self.data_args.experiment_mode == 'conditional_gen':
                    out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                    out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
                # if self.local_classes is not None:
                #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
                return arr, out_dict

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def _torch_collate_batch(examples, pad_token_id, max_length):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # length_of_first = examples[0].size(0)
    # Check if padding is necessary.
    # are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    # if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
    #     return torch.stack(examples, dim=0)
    # Creating the full tensor and filling it with our data.
    # max_length = max(x.size(0) for x in examples)
    # if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
    #     max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        if True:
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result