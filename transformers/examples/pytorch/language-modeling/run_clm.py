#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch
import datasets
import stanza
import spacy_stanza
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise, GPT2VAE, AR_for_cont,\
    Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree, Classifier_Consistency

from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.17.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False, pad_mask_id=None):
    if pad_mask_id is None:
        pad_mask_id = pad_token_id
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_mask_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    experiment: Optional[str] = field(
        default='compress',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    learned_emb: Optional[str] = field(
        default='no',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    padding_mode: Optional[str] = field(
        default='block',
        metadata={"help": "blcok or pad"},
    )
    roc_train: Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/ROCstory',
        metadata={"help": "roc story path"},
    )
    wiki_train: Optional[str] = field(
        default='/u/scr/xlisali/diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
        metadata={"help": "simple wiki path"},
    )
    e2e_train: Optional[str] = field(
        default='/u/scr/xlisali/e2e_data',
        metadata={"help": "simple wiki path"},
    )

    reduced_emb: Optional[int] = field(
        default=8,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rounding_mode: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    sigma: Optional[float] = field(
        default=1.0,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    n_embd: Optional[int] = field(
        default=16,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    init_emb: Optional[str] = field(
        default="",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    task: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    synth_config:  Optional[str] = field(
        default='/juice/scr/xlisali/diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def get_corpus_rocstory(data_args):
    '''

    :param data_args:  --> this is actually the model_args in the main function.
    :return:
    '''
    import csv, json
    from collections import Counter, defaultdict
    from spacy.lang.en import English
    import numpy as np

    # print(data_args.task, 'DEBUG', '*---'*100)
    # print(model_args.task, 'DEBUG', '*---' * 100)
    if data_args.experiment.startswith('roc') and data_args.task == 'infill':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                for ii in [1, 2, 3]:
                    sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [x.text for x in tokenizer(sent)]
                    sentence_lst.append(example)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('roc') and data_args.task == 'classifier':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/ROCstory_full.csv', 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for idx, row in enumerate(roc_reader):
                if idx == 0:
                    continue
                sentences = row[2:]
                sentences = [[x.text for x in tokenizer(sent)] for sent in sentences]
                for ii in [1, 2, 3]:
                    # sent = " ".join([sentences[ii-1], sentences[ii+1], sentences[ii]])
                    example = [sentences[ii-1], sentences[ii+1], sentences[ii], 1]
                    sentence_lst.append(example)
        np.random.shuffle(sentence_lst)

        # construct negative examples/
        wrong_lst = []
        for idx, sent in enumerate(sentence_lst[:-1]):
            wrong_mid = sentence_lst[idx+1][2]
            wrong_tup = (sent[0], sent[1], wrong_mid, 0)
            wrong_lst.append(wrong_tup)

        sentence_lst = sentence_lst + wrong_lst

        print(sentence_lst[:2], sentence_lst[-2:])
        return sentence_lst, {}




    elif data_args.experiment.startswith('roc') and data_args.task != 'data_teacher':
        print('loading dataset from ROCStory')
        nlp = English()
        tokenizer = nlp.tokenizer
        sentence_lst = []
        with open(f'{data_args.roc_train}/roc_train.json', 'r') as roc_reader:
            for row in roc_reader:
                sentences = json.loads(row)[0].strip()
        # with open(data_args.roc_train, 'r') as csvfile:
        #     roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
        #     for row in roc_reader:
        #         sentences = " ".join(row[2:])
                word_lst = [x.text for x in tokenizer(sentences)]
                sentence_lst.append(word_lst)
        # sentence_lst = sentence_lst[1:]
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('roc') and data_args.task == 'data_teacher':
        print('loading dataset from ROCStory')
        sentence_lst = []
        with open(data_args.roc_train, 'r') as csvfile:
            roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            for row in roc_reader:
                sentences = " ".join(row[2:])
                sentence_lst.append(sentences)
        sentence_lst = sentence_lst[1:]
        print(sentence_lst[:2])
        return sentence_lst, None
    elif data_args.experiment.startswith('simple-wiki'):
        print('loading dataset from simple wikipedia')
        sentence_lst = []
        with open(data_args.wiki_train, 'r') as ff:
            for row in ff:
                word_lst = row.split()
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])
    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'data_teacher':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'finetuneUNK':
        '''
            Used to evaluate fluency: first load e2e-vocab, and then UNK the oov words in the training data. 
        '''
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        # load vocab.
        tokenizer2 = load_tokenizer('e2e-tgt', 'random',
                                   '/u/scr/nlp/xlisali/predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart')
        vocab = {v: k for k, v in tokenizer2.items()}
        print(len(tokenizer2), len(vocab), 'loaded vocabs')

        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                tokenized = [x.text for x in tokenizer(word_lst)]
                word_lst1 = [x if x in vocab else 'UNK' for x in tokenized]
                word_lst1 = " ".join(word_lst1)
                word_lst2 = [vocab.get(x.text, vocab['UNK']) for x in tokenizer(word_lst)]
                word_lst2 = " ".join([tokenizer2[x] for x in word_lst2])
                # print(word_lst1, word_lst2)
                assert word_lst1 == word_lst2

                # print(word_lst1)
                sentence_lst.append(word_lst1)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt') and data_args.task == 'right2left':
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = list(reversed([x.text for x in tokenizer(word_lst)]))
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])

    elif data_args.experiment.startswith('e2e-tgt'):
        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        with open(path, 'r') as ff:
            for row in ff:
                word_lst = row.split('||')[1]
                word_lst = [x.text for x in tokenizer(word_lst)]
                sentence_lst.append(word_lst)
        print(sentence_lst[:2])


    elif data_args.experiment.startswith('e2e-back'):
        ordered_ = ['name', 'Type', 'area', 'customer rating', 'near',
                    'family friendly', 'food', 'price']
        full_dict = defaultdict(lambda:Counter())
        def ordered_fill(src_lst, mode='full', full_dict=None):
            pair_lst = {x.split(':')[0].lstrip().strip():x.split(':')[1].lstrip().strip() for x in src_lst.split('|')}
            # print(pair_lst, 'hello')
            if mode == 'full':
                for x in ordered_:
                    v = pair_lst.get(x, 'none')
                    result_lst.append(f"{x} : {v}")
                return "|".join(result_lst)
            else:
                # print(pair_lst)
                v = pair_lst.get(mode, 'none')
                full_dict[mode][v] += 1
                # print(v)
                return f"{mode} : {v}"

        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'{data_args.e2e_train}/src1_train.txt'
        vocab_lst = []
        with open(path, 'r') as ff:
            for row in ff:
                src_lst, word_lst = row.split('||')
                # src_lst = ordered_fill(src_lst, 'food')
                # src_lst = ordered_fill(src_lst, 'price')

                word_lst = [x.text for x in tokenizer(word_lst)]
                for mode in ordered_:
                    src_lst3 = ordered_fill(src_lst, mode, full_dict)
                    src_lst2 = [x.text for x in tokenizer(src_lst3)]
                    sentence_lst.append((word_lst, src_lst2))
                vocab_lst.append(word_lst)

                # src_lst = ordered_fill(src_lst, 'area')
                # word_lst = [x.text for x in tokenizer(word_lst)]
                # src_lst = [x.text for x in tokenizer(src_lst)]
                # sentence_lst.append((word_lst, src_lst))
        print(sentence_lst[:2])
        print(full_dict)

        counter = Counter()
        for input_ids in vocab_lst:
            counter.update(input_ids)
            # counter.update(src_ids)

    # get tokenizer.
    if not data_args.experiment.startswith('e2e-back'):
        counter = Counter()
        for input_ids in sentence_lst:
            counter.update(input_ids)

    vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
    for k, v in counter.items():
        if v > 10:
            vocab_dict[k] = len(vocab_dict)
    print(len(counter), len(vocab_dict))

    return sentence_lst, vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if model_args.experiment.startswith('synth'):
        import yaml, torch
        sys.path.insert(0, '/juice/scr/xlisali/diffusion_lm/synthetic_data/rnns-stacks')
        from dataset import Dataset as SynthDataset
        args_synth = yaml.load(open(data_args.synth_config))
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # args_synth['device'] = device
        print(args_synth)
        dataset_synth = SynthDataset(args_synth)
        print(dataset_synth.train_dataset[:5])
        from datasets import Dataset
        train_datasets = Dataset.from_dict({'text': dataset_synth.train_dataset})
        raw_datasets = datasets.DatasetDict()
        raw_datasets['train'] = train_datasets
        raw_datasets['validation'] = Dataset.from_dict({'text': dataset_synth.test_dataset})
        raw_datasets.vocab = dataset_synth.vocab
    elif model_args.experiment.startswith('pos'):
        import yaml, torch, json
        from collections import Counter, defaultdict
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, )
        dataset = dataset.load_from_disk('/u/scr/nlp/xlisali/wikitext-2-pos')
        counter = Counter()
        print(dataset)
        for input_ids in dataset['train']:
            counter.update(input_ids['pos'])
        print(counter)
        print(dataset)
        vocab_dict = {'START': 0, 'END': 1}
        for k in counter.keys():
            vocab_dict[k] = len(vocab_dict)

        dataset.vocab = vocab_dict
        from datasets import Dataset
        raw_datasets = dataset

    ###################### LOAD DATASETS & dictionary #########################
    elif model_args.experiment.startswith('roc') or\
            model_args.experiment.startswith('simple-wiki') or \
            model_args.experiment.startswith('e2e-tgt') or \
            model_args.experiment.startswith('e2e-back'):
        train_dataset, vocab = get_corpus_rocstory(model_args) # TODO: include validation sets.
        print(len(vocab), 'derived vocabs')

        if model_args.experiment.startswith('roc'):
            tokenizer = load_tokenizer('roc', 'random',
                                       '/u/scr/nlp/xlisali/predictability/diffusion_models_v7/diff_roc_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart')
            vocab = {v: k for k, v in tokenizer.items()}
            print(len(tokenizer), len(vocab), 'loaded vocabs')

        # train_dataset = train_dataset[:100]
        from datasets import Dataset

        if model_args.task == 'classifier':
            print(len(train_dataset))
            train_dataset = list(zip(*train_dataset))
            print(len(train_dataset))
            train_datasets = Dataset.from_dict({'left_text': train_dataset[0],
                                                'right_text':train_dataset[1],
                                                'mid_text':train_dataset[2],
                                                'label':train_dataset[3]})
        else:
            train_datasets = Dataset.from_dict({'text': train_dataset})
        raw_datasets = train_datasets.train_test_split(0.01)
        print(raw_datasets)

        if model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
            pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}
            pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                       'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                       'PUNCT', 'SYM', 'X']
            for x in pos_lst:
                pos_vocab[x] = len(pos_vocab)
        elif model_args.experiment in ['e2e-tgt-tree', 'e2e-tgt-gen-tree', 'e2e-tgt-gen-spans']:
            import benepar
            parser = benepar.Parser("benepar_en3")
            tree_vocab = parser._parser.config["label_vocab"]

        raw_datasets.vocab = vocab
        raw_datasets['validation'] = raw_datasets['test']

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    ############# LOAD TOKENIZER ##############
    if model_args.experiment.startswith('synth') or \
            model_args.experiment.startswith('pos') or model_args.experiment.startswith('roc') or \
            model_args.experiment.startswith('simple-wiki') or \
            model_args.experiment.startswith('e2e-tgt') or\
            model_args.experiment.startswith('e2e-back'):
        print('\ninitializing the tokenizer with small vocab\n' + '*'*100)

        if model_args.task in ['data_teacher', 'finetune']:
            print('loading from pretrained models tokenizer')
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
            print(type(tokenizer))
            if model_args.experiment == 'e2e-tgt-gen-tree' or model_args.experiment == 'e2e-tgt-gen-spans':
                # new_vocabs_added = list(tree_vocab.keys())
                tokenizer.add_tokens(list(tree_vocab.keys()))
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            elif model_args.experiment == 'e2e-tgt-gen-pos':
                tokenizer.add_tokens(list(pos_vocab.keys()))
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            elif model_args.experiment == 'e2e-tgt-gen-length':
                tokenizer.add_tokens([str(xx) for xx in range(64)])
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            elif model_args.experiment == 'e2e-tgt' and model_args.task == 'finetune':
                tokenizer.add_tokens(['UNK'])
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            else:
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        else:
            print('loading from dataset-specific vocab')
            tokenizer = raw_datasets.vocab
            if model_args.experiment == 'e2e-tgt-gen-spans':
                print('update the vocab to include tree vocabs')
                print(len(tokenizer))
                for x in tree_vocab.keys():
                    tokenizer[x] = len(tokenizer)
                print('update the vocab to include indices')
                # tokenizer.add_tokens([str(xx) for xx in range(64)])
                for x in range(64):
                    if str(x) not in tokenizer:
                        tokenizer[str(x)] = len(tokenizer)
            print(len(tokenizer))
            reverse_tokenizer = {v: k for k, v in tokenizer.items()}

    else:
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

    if model_args.model_name_or_path:
        if model_args.experiment == 'compress':
            config.reduced_emb = model_args.reduced_emb
            model = GPT2LMHeadModelCompress.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        elif model_args.experiment == 'bert_compress':
            config.reduced_emb = model_args.reduced_emb
            model = BERTModelCompress.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            # freeze parameters.
            if model_args.task == 'freeze':
                for param in model.base_model.parameters():
                    param.requires_grad = False

                for name, param in model.named_parameters():
                    # print(name)
                    if 'down_proj' in name or 'bert.embeddings.word_embeddings' in name or 'up_proj' in name:
                        print(name)
                        param.requires_grad = True
                        # total_params += param.numel()

        elif model_args.experiment == 'gpt2vae':
            config.reduced_emb = model_args.reduced_emb
            config.sigma = model_args.sigma

            model = GPT2VAE.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            # freeze parameters.
            if model_args.task == 'freeze':
                for param in model.transformer.parameters():
                    param.requires_grad = False

                for param in model.lm_head.parameters():
                    param.requires_grad = False

        elif model_args.experiment == 'autoencoder_noise':
            config.reduced_emb = model_args.reduced_emb
            config.rounding_mode = model_args.rounding_mode
            config.sigma = model_args.sigma

            if config.rounding_mode == 'gpt2':
                config2 = AutoConfig.from_pretrained('gpt2',)
                print(config2.is_decoder)
                config2.is_decoder = True
                config2.add_cross_attention = True
            else:
                config2 = None
            model = AutoEncoderWithNoise.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                config2=config2,
                cache_dir=model_args.cache_dir,
            )

        ############# LOAD MODELS for controllable generators ##############
        elif model_args.experiment in ['e2e-tgt-gen-tree', 'e2e-tgt-gen-pos', 'e2e-back-gen', 'e2e-tgt-gen-length',
                                       'e2e-tgt-gen-spans']:
            if model_args.task == 'finetune':
                model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
                model.resize_token_embeddings(len(tokenizer))
            elif model_args.task == 'from_scratch':
                config.vocab_size = len(tokenizer)
                print('\n Initializing the model from scratch \n' + '*' * 100)
                model = AutoModelForCausalLM.from_config(config)


        ############# LOAD MODELS for controllable classifier ##############
        elif model_args.experiment in ['e2e-back', 'e2e-back_t2', 'e2e-tgt-pos', 'e2e-tgt-tree']:
            import torch
            config.vocab_size = len(tokenizer)
            print('\n Initializing the model from scratch \n' + '*' * 100)

            # EDIT
            # also loading the diffusion model.
            import json, argparse
            from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
            config_path = os.path.join(model_args.init_emb, "training_args.json")
            print(config_path)
            # parser = argparse.ArgumentParser(description='Process some integers.')
            # args = parser.parse_args()
            with open(config_path, 'rb', ) as f:
                training_args2 = json.load(f)
            # args.__dict__.update(training_args)
            training_args2['sigma_small'] = True
            training_args2['diffusion_steps'] = 200  # 500  # DEBUG
            temp_dict = model_and_diffusion_defaults()
            temp_dict.update(training_args2)
            _, diffusion = create_model_and_diffusion(
                **temp_dict
            )

            config.input_emb_dim = model_args.n_embd
            config.train_diff_steps = training_args2['diffusion_steps']

            if model_args.experiment == 'e2e-back_t2':
                model = Classifier_Times(config=config, diffusion=diffusion,)
            elif model_args.experiment == 'e2e-back':
                model = Classifier_GPT2(config=config, diffusion=diffusion,)
            elif model_args.experiment == 'e2e-tgt-pos':
                config.pos_vocab_size = len(pos_vocab)
                model = Classifier_POS(config=config, diffusion=diffusion, )
            elif model_args.experiment == 'e2e-tgt-tree':
                config.tree_vocab_size = len(tree_vocab)
                print('tree vocab size', len(tree_vocab), '*'*20)
                model = Classifier_Tree(config=config, diffusion=diffusion, )



            filename = model_args.init_emb  # '/u/scr/nlp/xlisali/predictability/diffusion_models_v3/diff_e2e-tgt_block_rand16_transformer_lr0.0001_2000_cosine_Lsimple_h128_s2_sd101'
            path_save = '{}/random_emb.torch'.format(filename)
            path_learned = '{}/ema_0.9999_200000.pt'.format(filename)
            if model_args.experiment == 'e2e-tgt-pos' and model_args.learned_emb == 'no':
                model.transformer.embeddings.word_embeddings.load_state_dict(torch.load(path_save))
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment == 'e2e-tgt-pos' and model_args.learned_emb == 'yes':
                print('loading the learned embeddings')
                learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                model.transformer.embeddings.word_embeddings.weight.data = learned_embeddings.clone()
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment == 'e2e-tgt-tree' and model_args.learned_emb == 'no':
                model.transformer.embeddings.word_embeddings.load_state_dict(torch.load(path_save))
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment == 'e2e-tgt-tree' and model_args.learned_emb == 'yes':
                print('loading the learned embeddings')
                learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                model.transformer.embeddings.word_embeddings.weight.data = learned_embeddings.clone()
                model.transformer.embeddings.word_embeddings.weight.requires_grad = False
            elif model_args.experiment.startswith('e2e-back') and model_args.learned_emb == 'no':
                model.transformer.wte.load_state_dict(torch.load(path_save))
                model.transformer.wte.weight.requires_grad = False
            elif model_args.experiment.startswith('e2e-back') and model_args.learned_emb == 'yes':
                print('loading the learned embeddings')
                learned_embeddings = torch.load(path_learned)['word_embedding.weight']
                model.transformer.wte.weight.data = learned_embeddings.clone()
                model.transformer.wte.weight.requires_grad = False



        elif model_args.experiment in ['pos', 'synth', 'roc', 'simple-wiki', 'e2e-tgt']:

            if model_args.task == 'ar_for_cont':
                import torch
                config.sigma = model_args.sigma
                config.n_embd = model_args.n_embd
                config.n_head = model_args.n_embd
                config.vocab_size = len(tokenizer)
                model = AR_for_cont(config)
                filename =  model_args.init_emb #'/u/scr/nlp/xlisali/predictability/diffusion_models_v3/diff_e2e-tgt_block_rand16_transformer_lr0.0001_2000_cosine_Lsimple_h128_s2_sd101'
                path_save = '{}/random_emb.torch'.format(filename)
                model.transformer.wte.load_state_dict(torch.load(path_save))
                model.transformer.wte.weight.requires_grad = False
                print(model.lm_head.weight)
                print(model.transformer.wte.weight)
            if model_args.task == 'data_teacher' or model_args.task == 'finetune':
                print('\n FINETUNE THE MODEL.  \n' + '*' * 100)
                model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
                model.resize_token_embeddings(len(tokenizer))
            elif model_args.task == 'classifier':
                import torch
                config.vocab_size = len(tokenizer)
                config.type_vocab_size = 3 
                print('\n Initializing the model from scratch \n' + '*' * 100)
                import json, argparse
                from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
                config_path = os.path.join(model_args.init_emb, "training_args.json")
                print(config_path)
                with open(config_path, 'rb', ) as f:
                    training_args2 = json.load(f)
                # args.__dict__.update(training_args)
                training_args2['sigma_small'] = True
                training_args2['diffusion_steps'] = 200  # 500  # DEBUG
                temp_dict = model_and_diffusion_defaults()
                temp_dict.update(training_args2)
                _, diffusion = create_model_and_diffusion(**temp_dict)
                config.input_emb_dim = model_args.n_embd
                config.train_diff_steps = training_args2['diffusion_steps']
                model = Classifier_Consistency(config=config, diffusion=diffusion,)
                path_save = '/u/scr/nlp/xlisali/predictability/diffusion_models_v7/diff_roc_pad_rand128_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd108_xstart_e2e_long/model750000.pt'
                embedding_weight = torch.load(path_save)['word_embedding.weight']
                print(embedding_weight.shape)
                model.bert.embeddings.word_embeddings.weight = embedding_weight
                model.bert.embeddings.word_embeddings.weight.requires_grad = False
            else:
                config.vocab_size = len(tokenizer)
                print('\n Initializing the model from scratch \n' + '*' * 100)
                model = AutoModelForCausalLM.from_config(config)

        elif model_args.experiment in ['synth_emb', 'pos_emb', 'roc_emb', 'simple-wiki_emb',
                                       'e2e-tgt_emb']:
            print(f'\n Initializing the model from scratch with dim {model_args.reduced_emb} \n'
                  + '*' * 100)
            config.reduced_emb = model_args.reduced_emb
            model = GPT2LMHeadModelCompress(
                config=config,
            )

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if model_args.experiment.startswith('synth'):
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                result_dict = {'input_ids': examples['text'], 'labels': examples['text']}
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            lm_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif model_args.experiment in ['e2e-tgt-pos', 'e2e-tgt-gen-pos']:
        assert model_args.task != 'data_teacher', 'should not be data_teacher.'
        # nlp = stanza.Pipeline(lang='en', processors='mwt,pos')
        nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                sent_lst = [" ".join(seq) for seq in examples['text']]
                sent_full = " ".join(sent_lst)
                doc = nlp(sent_full)
                doc_token_pos = [(token.text, token.pos_,) for token in doc]
                len_lst = [len(seq) for seq in examples['text']]
                # print(sum(len_lst),  len(doc_token_pos))
                assert sum(len_lst) == len(doc_token_pos)
                pos_lst = []
                init_idx = 0
                for len_temp in len_lst:
                    pos_lst.append([x[1] for x in doc_token_pos[init_idx:init_idx+len_temp]])
                    init_idx = init_idx+len_temp

                if model_args.experiment == 'e2e-tgt-pos':
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
                    print(pos_tags)
                    result_dict = {'input_ids': input_ids, 'pos_tags':pos_tags}
                elif model_args.experiment == 'e2e-tgt-gen-pos':
                    if model_args.task == 'finetune':
                        input_strings = [" ".join(pos_) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                                         for (pos_, seq) in zip(pos_lst, examples['text'])]
                        return tokenizer(input_strings, max_length=128, padding='max_length', truncation=True)
                    elif model_args.task == 'from_scratch':
                        input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                        pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
                        result_dict = {'input_ids': input_ids, 'pos_tags': pos_tags}

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            if model_args.experiment == 'e2e-tgt-pos':
                vocab_dict = raw_datasets.vocab
                max_length = 64
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                max_src_length = 64 #min(seqlen, max_src_length)
                group_lst['pos_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['pos_tags'],
                                                                                    vocab_dict['PAD'], max_src_length, return_mask=True)
                group_lst['labels'] = [[-100] * len(x) + y for (x, y) in zip(group_lst['input_ids'], group_lst['pos_ids'])]
            elif model_args.experiment == 'e2e-tgt-gen-pos':
                group_lst['labels'] = group_lst['input_ids']

            return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"padding",
            )

    elif model_args.experiment in ['e2e-tgt-gen-length']:
        assert model_args.task != 'data_teacher', 'should not be data_teacher.'
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                if model_args.task == 'finetune':
                    input_strings = [f'{len(seq)}' + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token
                                     for seq in examples['text']]
                    return tokenizer(input_strings, max_length=128, padding='max_length', truncation=True)
                elif model_args.task == 'from_scratch':
                    raise NotImplementedError
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    pos_tags = [[0] + [pos_vocab[x] for x in seq] + [1] for seq in pos_lst]
                    result_dict = {'input_ids': input_ids, 'pos_tags': pos_tags}

            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            group_lst['labels'] = group_lst['input_ids']
            return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"padding",
            )

    elif model_args.experiment in ['e2e-tgt-tree', 'e2e-tgt-gen-tree', 'e2e-tgt-gen-spans']:
        assert model_args.task != 'data_teacher', 'should not be data_teacher.'
        import spacy, nltk
        from nltk.tree import Tree
        import numpy as np
        import torch
        # print(parser)
        print(parser._parser.config["label_vocab"])
        from utils import chart_from_tree, remove_leaves, pad_charts

        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                sent_lst = []
                for sent in examples['text']:
                    # print(sent)
                    input_sentence1 = benepar.InputSentence(
                        words=sent[:63],
                    )
                    sent_lst.append(input_sentence1)
                parse_lst = list(parser.parse_sents(sent_lst))
                assert len(parse_lst) == len(examples['text'])

                if model_args.experiment == 'e2e-tgt-tree':
                    chart_lst = []
                    for x in parse_lst:
                        chart = chart_from_tree(tree_vocab, x)
                        chart_lst.append(chart)
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    result_dict = {'input_ids': input_ids, 'chart_lst':chart_lst}
                elif model_args.experiment == 'e2e-tgt-gen-tree':
                    parse_lst = [remove_leaves(tree) for tree in parse_lst]

                    if model_args.task == 'finetune':
                        input_strings = [tree._pformat_flat("", "()", False) + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token for
                                         (tree, seq) in zip(parse_lst, examples['text'])]
                        return tokenizer(input_strings, max_length=256, padding='max_length', truncation=True)
                    elif model_args.task == 'from_scratch':
                        raise NotImplementedError
                        input_ids = [tree + [0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (tree, seq) in
                                     zip(parse_lst, examples['text'])]
                elif model_args.experiment == 'e2e-tgt-gen-spans':
                    if model_args.task == 'finetune':
                        input_strings = []
                        for parse, seq in zip(parse_lst, examples['text']):
                            chart, spans = chart_from_tree(tree_vocab, parse, verbose=True)
                            for (a,b,c) in spans:
                                input_strings.append(f"{a}, {b}, {c}" + tokenizer.bos_token + " ".join(seq) + tokenizer.eos_token )
                        # print(len(input_strings), len(examples['text']))
                        return tokenizer(input_strings, max_length=70, padding='max_length', truncation=True)
                    elif model_args.task == 'from_scratch':
                        input_lst = []
                        for parse, seq in zip(parse_lst, examples['text']):
                            chart, spans = chart_from_tree(tree_vocab, parse, verbose=True)
                            for (a, b, c) in spans:
                                input_ids = [vocab_dict.get(x, vocab_dict['UNK']) for x in f"{a} {b} {c}".split()] + [0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1]
                                input_lst.append(input_ids)
                        print(len(input_lst), len(parse_lst))
                        print(input_lst[0])
                        result_dict = {'input_ids': input_lst}


            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            vocab_dict = raw_datasets.vocab
            if model_args.experiment == 'e2e-tgt-tree':
                max_length = 64
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                group_lst['parse_chart'] = pad_charts(group_lst['chart_lst'], padding_value=-100)
            elif model_args.experiment == 'e2e-tgt-gen-tree' or  model_args.experiment == 'e2e-tgt-gen-spans':
                if model_args.task == 'finetune':
                    group_lst['labels'] = group_lst['input_ids']
                elif model_args.task == 'from_scratch':
                    max_length = 64
                    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'],
                                                                   max_length)
                    group_lst['labels'] = group_lst['input_ids']

            return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"padding",
            )

    elif (model_args.experiment.startswith('roc') or\
            model_args.experiment.startswith('simple-wiki') or \
            model_args.experiment.startswith('e2e-tgt')) and model_args.task not in ['data_teacher', 'finetune']:
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                if model_args.task == 'classifier':

                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq1 + seq2 + seqmid] + [1] for (seq1, seq2, seqmid) in
                                 zip(examples['left_text'], examples['right_text'], examples['mid_text'])]

                    type_ids = [[0] + [0] * (len(seq1)+len(seq2)) + [1] * len(seqmid) + [1] for
                                 (seq1, seq2, seqmid) in
                                 zip(examples['left_text'], examples['right_text'], examples['mid_text'])]

                    labels = examples['label']
                    result_dict = {'input_ids': input_ids, 'type_ids':type_ids, 'labels':labels}
                else:
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
                    result_dict = {'input_ids': input_ids}
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if model_args.padding_mode == 'block':
            block_size = 64
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
                result["labels"] = result["input_ids"].copy()
                return result

            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
        else:
            def pad_function(group_lst):
                vocab_dict = raw_datasets.vocab

                max_length = 64
                seqlen = 64
                if model_args.task == 'classifier':
                    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'],
                                                                   max_length)
                    group_lst['type_ids'] = _collate_batch_helper(group_lst['type_ids'], 2,
                                                                   max_length)
                    group_lst["labels"] = group_lst["labels"]
                else:
                    group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
                    group_lst["labels"] = group_lst["input_ids"].copy()

                return group_lst

            with training_args.main_process_first(desc="grouping texts together"):
                lm_datasets = tokenized_datasets.map(
                    pad_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"padding",
                )

    elif model_args.experiment.startswith('e2e-back'):
        def tokenize_function(examples):
            vocab_dict = raw_datasets.vocab
            with CaptureLogger(tok_logger) as cl:
                if model_args.experiment == 'e2e-back':
                    input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (seq, _) in examples['text']]
                    src_ids = [ [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for (_, seq) in examples['text']]
                    result_dict = {'word_ids': input_ids, 'src_ids':src_ids}
                elif model_args.experiment == 'e2e-back-gen':
                    input_strings = [
                        " ".join(attributes) + tokenizer.bos_token + " ".join(words) + tokenizer.eos_token
                        for (words, attributes) in examples['text']]
                    return tokenizer(input_strings, max_length=100, padding='max_length', truncation=True)
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            if model_args.experiment == 'e2e-back':
                vocab_dict = raw_datasets.vocab
                max_length = 64
                seqlen = 64
                group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
                max_src_length = max([len(xx) for xx in group_lst['src_ids']])
                # print(max_src_length, seqlen)
                max_src_length = min(seqlen, max_src_length)
                group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                                    vocab_dict['PAD'],
                                                                                    max_src_length,
                                                                                    return_mask=True)

                group_lst['input_ids'] = [x + y  for (x,y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
                group_lst['labels'] = [[-100] * len(x) + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
            elif model_args.experiment == 'e2e-back-gen':
                group_lst['labels'] = group_lst['input_ids']
            return group_lst

        # def pad_function2(group_lst):
        #     vocab_dict = raw_datasets.vocab
        #
        #     max_length = 64
        #     seqlen = 64
        #     group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        #     max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        #     # print(max_src_length, seqlen)
        #     max_src_length = min(seqlen, max_src_length)
        #     group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
        #                                                                         vocab_dict['PAD'],
        #                                                                         max_src_length,
        #                                                                         return_mask=True)
        #
        #     group_lst['input_ids'] = group_lst['word_ids']
        #     group_lst['tgt_ids'] = group_lst['src_ids']
        #     group_lst['labels'] = [[-100] * (len(x) * 2) + y for (x, y) in zip(group_lst['word_ids'], group_lst['src_ids'])]
        #
        #     return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function, #if model_args.experiment == 'e2e-back' else pad_function2,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    elif model_args.experiment.startswith('pos'):
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                input_ids = [[0] + [raw_datasets.vocab[x] for x in seq]+ [1] for seq in examples['pos']]
                # input_ids = [0] + input_ids + [1]
                result_dict = {'input_ids': input_ids}
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return result_dict

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        block_size = 64
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
            result["labels"] = result["input_ids"].copy()
            return result

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    elif (model_args.experiment.startswith('roc') or\
            model_args.experiment.startswith('simple-wiki') or \
            model_args.experiment.startswith('e2e-tgt')) and model_args.task in ['data_teacher', 'finetune']:
        print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                # print([x for x in  examples[text_column_name]])
                output = tokenizer([tokenizer.bos_token + x + tokenizer.eos_token for x in examples[text_column_name]])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        def pad_function(group_lst):
            max_length = 100
            group_lst['input_ids'], group_lst['attention_mask'] = _collate_batch_helper(group_lst['input_ids'],
                                                                                        tokenizer.pad_token_id,
                                                                                        max_length, return_mask=True,
                                                                                        pad_mask_id=0)
            for x in group_lst['input_ids']:
                assert len(x) == max_length
            group_lst["labels"] = group_lst["input_ids"].copy()

            return group_lst

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                pad_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"padding",
            )

    else:
        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(examples[text_column_name])
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx."
                )
                block_size = 1024
        else:
            if data_args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            print(logits[0].shape, logits[1].shape)
            if type(logits) == tuple:
                return logits[0].argmax(dim=-1)
            else:
                return logits.argmax(dim=-1)

        metric = load_metric("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    trainer_tokenizer = None if ((model_args.experiment in ['pos', 'synth', 'roc', 'simple-wiki', 'e2e-tgt',
                                                            'e2e-tgt-pos','e2e-tgt-tree', 'e2e-back', 'e2e-back_t2']
                                 or model_args.experiment in ['synth_emb', 'pos_emb', 'roc_emb', 'simple-wiki_emb', 'e2e-tgt_emb'])
                                 and model_args.task not in ['data_teacher', 'finetune']) \
                        else tokenizer
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=trainer_tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        # compute_metrics=compute_metrics if training_args.do_eval else None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
