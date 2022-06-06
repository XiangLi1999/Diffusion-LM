#! /usr/bin/env python3
# coding=utf-8

# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
import math
import time, os
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.rounding import rounding_func, load_models

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch import nn
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange

from pplm_classification_head import ClassificationHead
from collections import defaultdict, Counter
from spacy.lang.en import English
from transformers import GPT2LMHeadModel, GPT2Tokenizer


torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "START Browns Cambridge is a moderately priced fast food restaurant . It has a customer rating of 3 out of 5 . \n END PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD"
#"This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def get_vocab_dict():
    model_path2 = 'predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart/ema_0.9999_200000.pt'
    # load configurations.
    config_path = os.path.join(os.path.split(model_path2)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args2 = json.load(f)
    training_args2['sigma_small'] = True
    training_args2['diffusion_steps'] = 200  # 500  # DEBUG
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update(training_args2)
    _, diffusion = create_model_and_diffusion(
        **temp_dict
    )


    model_embs, tokenizer = load_models(training_args2['modality'], training_args2['experiment'],
                                        training_args2['model_name_or_path'],
                                        training_args2['in_channel'],
                                        os.path.split(model_path2)[0])

    rev_tokenizer = {v:k for k, v in tokenizer.items()}
    return rev_tokenizer

class Discriminator(nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, class_size, pretrained_model="gpt2-medium", cached_mode=False, device="cpu"):
        super().__init__()
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        except:
            self.tokenizer = None
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
        hidden = self.encoder.transformer(x)["last_hidden_state"]
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = nn.functional.log_softmax(logits, dim=-1)

        return probs


class GeneratorDiscrim(nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(self, class_size, pretrained_model="gpt2-medium", cached_mode=False, device="cpu"):
        super().__init__()
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        except:
            self.tokenizer = None
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(class_size=class_size, embed_size=self.embed_size)
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()
        hidden = self.encoder.transformer(x)["last_hidden_state"]
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = nn.functional.log_softmax(logits, dim=-1)

        return probs

class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(len(sequences), max(lengths)).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer, epoch=0, log_interval=10, device="cpu"):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = nn.functional.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far,
                    len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset),
                    loss.item(),
                )
            )


def evaluate_performance(data_loader, discriminator, device="cpu"):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += nn.functional.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset), 100.0 * correct / len(data_loader.dataset)
        )
    )


def predict(input_sentence, model, classes, cached=False, device="cpu"):
    if isinstance(model.tokenizer, dict):
        input_t = [model.tokenizer[x] for x in input_sentence.split()]
    else:
        input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print(
        "Predictions:",
        ", ".join("{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in zip(classes, log_probs)),
    )


def get_cached_data_loader(dataset, batch_size, discriminator, shuffle=False, device="cpu"):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys), batch_size=batch_size, shuffle=shuffle, collate_fn=cached_collate_fn
    )

    return data_loader


def train_discriminator(
    dataset,
    dataset_fp=None,
    pretrained_model="gpt2-medium",
    epochs=10,
    batch_size=64,
    log_interval=10,
    save_model=False,
    cached=False,
    no_cuda=False,
    mode='name',
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "SST":
        idx2class = ["positive", "negative", "very positive", "very negative", "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        print(torchtext_data)
        label = torchtext_data.Field(sequential=False)
        text = torchtext_data.Field()
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        x = []
        y = []
        for i in trange(len(train_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(vars(train_data[i])["text"])
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in trange(len(test_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(vars(test_data[i])["text"])
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }

    elif dataset == "clickbait":
        idx2class = ["non_clickbait", "clickbait"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            data = []
            for i, line in enumerate(f):
                try:
                    data.append(eval(line))
                except Exception:
                    print("Error evaluating line {}: {}".format(i, line))
                    continue
        x = []
        y = []
        with open("datasets/clickbait/clickbait_train_prefix.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                    else:
                        print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                        continue
                    x.append(seq)
                    y.append(d["label"])
                except Exception:
                    print("Error evaluating / tokenizing" " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 1,
        }

    elif dataset == "toxic":
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        x = []
        y = []
        with open("datasets/toxic/toxic_train.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])

                    if len(seq) < max_length_seq:
                        seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)
                    else:
                        print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                        continue
                    x.append(seq)
                    y.append(int(np.sum(d["label"]) > 0))
                except Exception:
                    print("Error evaluating / tokenizing" " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }
    elif dataset == 'e2e-name':
        ordered_ = ['name', 'Type', 'area', 'customer rating', 'near',
                    'family friendly', 'food', 'price']
        full_dict = defaultdict(lambda: Counter())

        def ordered_fill(src_lst, mode='full', full_dict=None):
            pair_lst = {x.split(':')[0].lstrip().strip(): x.split(':')[1].lstrip().strip() for x in src_lst.split('|')}
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
                return v

        print('loading dataset from simple e2e dataset')
        sentence_lst = []
        nlp = English()
        tokenizer = nlp.tokenizer
        path = f'e2e_data/src1_train.txt'
        vocab_lst = []
        with open(path, 'r') as ff:
            for row in ff:
                src_lst, word_lst = row.split('||')
                # src_lst = ordered_fill(src_lst, 'food')
                src_lst3 = ordered_fill(src_lst, mode, full_dict)
                # src_lst2 = [x.text for x in tokenizer(src_lst3)]
                word_lst = [x.text for x in tokenizer(word_lst)]
                sentence_lst.append((word_lst, src_lst3))

                # for mode in ordered_:
                #     src_lst3 = ordered_fill(src_lst, mode, full_dict)
                #     src_lst2 = [x.text for x in tokenizer(src_lst3)]
                #     sentence_lst.append((word_lst, src_lst2))
                vocab_lst.append(word_lst)

        print(sentence_lst[:2])
        print(full_dict)

        idx2class = [x for x in full_dict[mode]]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        vocab_dict = get_vocab_dict()
        discriminator.tokenizer = vocab_dict

        # print(vocab_dict)
        group_lst = {}
        group_lst['word_ids'] = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in text] + [1] for (text, _) in
                                 sentence_lst]
        group_lst['labels'] = [class2idx[tgt] for (_, tgt) in sentence_lst]

        max_length = 64
        group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        x = [torch.LongTensor(sent) for sent in group_lst['word_ids']]
        y = group_lst['labels']

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }
        pass


    else:  # if dataset == "generic":
        # This assumes the input dataset is a TSV with the following structure:
        # class \t text

        if dataset_fp is None:
            raise ValueError("When generic dataset is selected, " "dataset_fp needs to be specified aswell.")

        classes = set()
        with open(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for row in tqdm(csv_reader, ascii=True):
                if row:
                    classes.add(row[0])

        idx2class = sorted(classes)
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class), pretrained_model=pretrained_model, cached_mode=cached, device=device
        ).to(device)

        x = []
        y = []
        with open(dataset_fp) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(tqdm(csv_reader, ascii=True)):
                if row:
                    label = row[0]
                    text = row[1]

                    try:
                        seq = discriminator.tokenizer.encode(text)
                        if len(seq) < max_length_seq:
                            seq = torch.tensor([50256] + seq, device=device, dtype=torch.long)

                        else:
                            print("Line {} is longer than maximum length {}".format(i, max_length_seq))
                            continue

                        x.append(seq)
                        y.append(class2idx[label])

                    except Exception:
                        print("Error tokenizing line {}, skipping it".format(i))
                        pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    end = time.time()
    print("Preprocessed {} data points".format(len(train_dataset) + len(test_dataset)))
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(train_dataset, batch_size, discriminator, shuffle=True, device=device)

        test_loader = get_cached_data_loader(test_dataset, batch_size, discriminator, device=device)

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    if save_model:
        print('saving to', "{}_{}_classifier_head_meta.json".format(dataset, mode))
        with open("{}_{}_classifier_head_meta.json".format(dataset, mode), "w") as meta_file:
            json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device,
        )
        evaluate_performance(data_loader=test_loader, discriminator=discriminator, device=device)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class, cached=cached, device=device)

        if save_model and epoch == epochs-1:
            print('saving to', "{}_{}_classifier_head_epoch_{}.pt".format(dataset, mode, epoch + 1),)
            # torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch + 1
            #               ))
            torch.save(
                discriminator.get_classifier().state_dict(),
                "{}_{}_classifier_head_epoch_{}.pt".format(dataset, mode, epoch + 1),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SST",
        choices=("SST", "clickbait", "toxic", "generic", 'e2e-name'),
        help="dataset to train the discriminator on."
        "In case of generic, the dataset is expected"
        "to be a TSBV file with structure: class \\t text",
    )
    parser.add_argument(
        "--dataset_fp",
        type=str,
        default="",
        help="File path of the dataset to use. " "Needed only in case of generic datadset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="name",
        help="File path of the dataset to use. " "Needed only in case of generic datadset",
    )
    parser.add_argument(
        "--pretrained_model", type=str, default="gpt2-medium", help="Pretrained model to use as encoder"
    )
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="Number of training epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save_model", action="store_true", help="whether to save the model")
    parser.add_argument("--cached", action="store_true", help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true", help="use to turn off cuda")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
