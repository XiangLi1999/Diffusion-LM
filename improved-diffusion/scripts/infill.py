"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json, sys
import stanza
import spacy_stanza
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English

def main():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    if args.eval_task_.startswith('control_'):
        args.diffusion_steps = 200  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)
    logger.log('load the partial sequences')
    if args.partial_seq:
        partial_seq = [args.partial_seq]
        partial_seq_idx = ['0']
    elif args.partial_seq_file:
        # implies that we should read from the files
        nlp = English()
        tokenizer_spacy = nlp.tokenizer
        print(f'reading from the file {args.partial_seq_file}', '-*'*20)
        with open(args.partial_seq_file, 'r') as f:
            sent_lst = json.load(f)
        partial_seq = []
        partial_seq_idx = []
        for idx, (key, val) in enumerate(sent_lst.items()):
            if idx < int(args.start_idx) or idx > int(args.end_idx):
                continue
            partial_seq_ = f"{val['obs1']} " + "PAD " * 10 + f"{val['obs2']}"
            word_lst = [x.text for x in tokenizer_spacy(partial_seq_)]
            partial_seq_ = " ".join(word_lst)
            print(partial_seq_, idx)
            partial_seq.append(partial_seq_)
            partial_seq_idx.append(str(idx))
    else:
        partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                       'Alimentum , situated by the river , is quite child friendly .']
        partial_seq_idx = ['0', '1']
    # else:  generate them by randomly preturbing the inputs data.
    if args.modality in ['synth', 'pos']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = tokens2id['END']
        print(f'pad token = {todo_pad_token}')
        encoded_partial_seq = [th.LongTensor([tokens2id[x] for x in seq.split()]) for seq in partial_seq]
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    elif args.modality in ['e2e-tgt', 'roc', 'roc-aug']:
        tokens2id = {v:k for k, v in tokenizer.items()}
        todo_pad_token = -1
        pad_token = tokens2id['PAD']
        encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in partial_seq]
        if args.eval_task_ == 'infill':
            todo_pad_token = tokens2id['PAD']
            print(f'pad token = {todo_pad_token}')
            partial_seq = [(b, a) for (a,b) in zip(partial_seq, partial_seq_idx)]
            pass
        elif args.eval_task_ == 'l2r':
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            right_length= args.tgt_len - len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([seq, right_pad], dim=0)  for seq in encoded_partial_seq ]

        elif args.eval_task_ == 'r2l':
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            # right_length= args.image_size ** 2 - len(encoded_partial_seq[0])
            right_length= args.tgt_len - len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([right_pad, seq], dim=0)  for seq in encoded_partial_seq ]

        elif args.eval_task_ == 'length':
            right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
            # right_length = args.tgt_len - len(encoded_partial_seq[0])
            # assert args.tgt_len > len(encoded_partial_seq[0])
            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
            encoded_partial_seq[0][args.tgt_len-1] = tokens2id['END']
            encoded_partial_seq[0][args.tgt_len] = tokens2id['START']
            # encoded_partial_seq[0][args.tgt_len+1:] = tokens2id['PAD']

        elif args.eval_task_ == 'word':
            right_length= args.tgt_len // 2

            right_pad = th.empty(right_length).fill_(todo_pad_token).long()
            encoded_partial_seq = [th.cat([right_pad, seq, right_pad], dim=0)  for seq in encoded_partial_seq ]
        elif args.eval_task_.startswith('control'):
            # right_pad = th.empty(args.tgt_len+2).fill_(pad_token).long()
            # TO FIX... IMPORTANT.
            if 'length' not in args.eval_task_:
                right_pad = th.empty(64).fill_(pad_token).long()
                encoded_partial_seq = [th.cat([right_pad], dim=0)]
                encoded_partial_seq[0][0] = tokens2id['START']
                encoded_partial_seq[0][args.tgt_len] = tokens2id['END']

            if args.eval_task_ == 'control_attribute':
                model_control = Classifier_GPT2.from_pretrained('predictability/diff_models/e2e-back_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_full_multi16_t_aware').cuda()

                control_label_lst = []
                with open('diffusion_lm/improved-diffusion/control_gen/target_attribute.json', 'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:5])
                control_constraints = []
                for label_class in control_label_lst:
                    # assert encoded_partial_seq[0].size(0)  == 64
                    label = [-100] * 64 + [tokens2id.get(x, tokens2id['UNK']) for x in label_class]
                    label_ids = th.tensor(label).unsqueeze(0)
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn3, debug_lst, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, label_class))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-'*20)
                # # label_class =  ['price', ':', 'cheap']
                # # label_class =  ['name', ':', 'The', 'Vaults']
                # # label_class =  ['food', ':', 'French']
                # label_class = ['price', ':', 'none']
                # label_class = ['near', ':', 'riverside']
                # label_class = ['name', ':','The', 'Vaults'] #98%
                # label_class = ['name', ':', 'The', 'Cricketers'] #92%
                # label_class = ['name', ':','Green', 'Man'] #96%
                # label_class = ['price', ':', 'cheap'] #84%
                # label_class = ['price', ':', 'moderate'] #78%
                # label_class = ['area', ':', 'riverside'] #90%
                # label_class = ['UNK', ':', 'coffee', 'shop']#98%
                # label_class = ['UNK', ':', 'pub']  # 90%
                # label_class = ['name', ':', 'The', 'Rice', 'Boat'] # 100%
                # label_class = ['food', ':', 'French'] #82%
                # label_class = ['customer', 'rating', ':', 'average'] #78%
                # label_class = ['customer', 'rating', ':', '3', 'out', 'of', '5']  # 0.54
                # label_class = ['customer', 'rating', ':', '5', 'out', 'of', '5']  # 0.68
                # label_class = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                # # label_class = ['price', ':', 'less', 'than', '£', '20'] # 90%
                # # label_class = ['price', ':', 'cheap']  # 84%
                # label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class]
                # label_ids = th.tensor(label).unsqueeze(0)
                #
                # debug_lst = []
                # langevin_fn_selected = partial(langevin_fn3, debug_lst, model_control, model3.cuda(),
                #                                label_ids.expand(args.batch_size, -1), 0.1)
                # # langevin_fn_selected = partial(langevin_fn1, debug_lst, model_control, model3.cuda(),
                # #                                label_ids.expand(args.batch_size, -1), 0.1)

            if args.eval_task_ == 'control_attribute_compose':
                model_control = Classifier_GPT2.from_pretrained('predictability/diff_models/e2e-bac'
                                                                'k_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_'
                                                                'full_multi16_t_aware').cuda()
                label_ids_lst = []

                label_class = ['price', ':', 'none']
                label_class = ['near', ':', 'riverside']
                label_class = ['name', ':','The', 'Vaults'] #98%
                label_class = ['name', ':', 'The', 'Cricketers'] #92%
                label_class = ['name', ':','Green', 'Man'] #96%
                label_class = ['price', ':', 'cheap'] #84%
                label_class = ['price', ':', 'moderate'] #78%
                label_class = ['area', ':', 'riverside'] #90%
                label_class = ['UNK', ':', 'coffee', 'shop']#98%
                label_class = ['UNK', ':', 'pub']  # 90%
                label_class = ['name', ':', 'The', 'Rice', 'Boat'] # 100%
                label_class = ['food', ':', 'French'] #82%
                label_class = ['customer', 'rating', ':', 'average'] #78%
                label_class = ['customer', 'rating', ':', '3', 'out', 'of', '5']  # 0.54
                # label_class = ['customer', 'rating', ':', '5', 'out', 'of', '5']  # 0.68
                label_class1 = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                # label_class1 = ['price', ':', 'less', 'than', '£', '20'] # 90%
                # label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class1]
                # label_ids = th.tensor(label).unsqueeze(0).cuda()
                # label_ids_lst = [label_ids]

                label_ids_lst = []
                label_class2 = ['name', ':', 'Green', 'Man']  # 96% --> 82%
                label_class1 = ['price', ':', 'less', 'than', '£', '20']  # 90%
                for label_class in [label_class1, label_class2]:
                    label = [-100] * encoded_partial_seq[0].size(0) + [tokens2id[x] for x in label_class]
                    label_ids = th.tensor(label).unsqueeze(0).cuda()
                    label_ids_lst.append(label_ids)

                debug_lst = []
                langevin_fn_selected = partial(langevin_fn3_compose, debug_lst, model_control, model3.cuda(),
                                               [label_ids.expand(args.batch_size, -1) for label_ids in label_ids_lst],
                                               0.1)

            elif args.eval_task_ == 'control_pos':
                model_control = Classifier_POS.from_pretrained('predictability/diff_models/e2e-tgt-pos_e=6_b=10_m=bert-'
                                                               'base-uncased_wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()


                pos_vocab = {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3}

                pos_lst = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB',
                           'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ',
                           'PUNCT', 'SYM', 'X']
                for x in pos_lst:
                    pos_vocab[x] = len(pos_vocab)
                pos_vocab_rev = {v:k for k,v in pos_vocab.items()}

                ################33
                control_label_lst = []
                with open('diffusion_lm/improved-diffusion/control_gen/target_pos.json', 'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                print(control_label_lst[:5])
                control_constraints = []
                for label_class_dict in control_label_lst[:50]:#control_label_lst[:100]:
                    label_class = label_class_dict['pos']
                    words_ = label_class_dict['words_']
                    label_class = [pos_vocab.get(x, pos_vocab['UNK']) for x in label_class]
                    label_class = label_class + [pos_vocab['PAD']] * (64 - len(label_class))
                    label_ids = th.LongTensor(label_class).unsqueeze(0)
                    debug_lst = []
                    langevin_fn_selected = partial(langevin_fn4, debug_lst, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1),
                                                   0.1)
                    control_constraints.append((langevin_fn_selected, label_class_dict['pos']))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

                # toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.',
                #        'Prices', 'start', 'at', '£', '30', '.', '\n', 'END']
                # # toy1 = 'START The Mill is a coffee shop with an expensive menu near The Sorrento . \n END'.split()
                # toy = toy1 + (64 - len(toy1)) * ['PAD']
                # input_ids = th.tensor([tokens2id[x] for x in toy]).unsqueeze(0)
                #
                # model_out = model_control(input_ids.to(model_control.device), t=200)
                # print(model_out.logits.shape)
                # pred_pos = th.argmax(model_out.logits, dim=-1)
                # print(pred_pos.shape, pred_pos)
                # print('predicted pos', [pos_vocab_rev[x.item()] for x in pred_pos[0]])
                # model_out = model_control(input_ids.to(model_control.device), pos_ids=pred_pos, t=200)
                # print('predicted score', model_out.loss)
                #
                # nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
                # sent_full = " ".join(toy1[1:-1])
                # doc = nlp(sent_full)
                # doc_token_pos = [(token.text, token.pos_,) for token in doc]
                # print(doc_token_pos)
                # doc_token_pos = ['START'] + [x[1] for x in doc_token_pos] + ['END']
                # print(doc_token_pos, 'target POS tagging sequences')
                # label_class = [pos_vocab.get(x, pos_vocab['UNK']) for x in doc_token_pos]
                # label_class = label_class + [pos_vocab['PAD']] * (encoded_partial_seq[0].size(0)-len(label_class))
                # print(label_class)
                # label_ids = th.LongTensor(label_class).unsqueeze(0)
                # label_ids[:, 3:] = -100
                # label_ids[:, :1] = -100
                #
                # debug_lst = []
                # langevin_fn_selected = partial(langevin_fn4, debug_lst, model_control, model3.cuda(),
                #                                label_ids.expand(args.batch_size, -1),
                #                                0.1)

            elif args.eval_task_ == 'control_tree':
                # model_control = Classifier_Tree.from_pretrained(
                #     'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
                #     'wikitext-103-raw-v1_101_wp_full_multi16_v2').cuda()
                model_control = Classifier_Tree.from_pretrained(
                    'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
                    'wikitext-103-raw-v1_101_wp_full_multi16_cat').cuda()

                # print(model_control)

                import benepar
                from tree_helper import chart_from_tree, pad_charts, padded_chart_from_spans
                parser = benepar.Parser("benepar_en3")
                tree_vocab = parser._parser.config["label_vocab"]
                tree_vocab_rev = {v: k for k, v in tree_vocab.items()}

                ###############
                control_label_lst = []
                with open('diffusion_lm/improved-diffusion/control_gen/target_tree.json',
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:1])
                control_constraints = []
                for label_class_dict in control_label_lst[100:]:
                    padded_chart = th.LongTensor(label_class_dict['padded_chart'])
                    words_ = label_class_dict['words_']
                    label_ids = padded_chart
                    langevin_fn_selected = partial(langevin_fn_tree, 0.0005, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1, -1),
                                                   0.1)
                    control_constraints.append((langevin_fn_selected, [label_class_dict['tree']]))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)

            elif args.eval_task_ == 'control_span':
                model_control = Classifier_Tree.from_pretrained(
                    'predictability/diff_models/e2e-tgt-tree_e=20_b=32_m=bert-base-uncased_'
                    'wikitext-103-raw-v1_101_wp_full_multi16_cat').cuda()

                import benepar
                from tree_helper import chart_from_tree, pad_charts, padded_chart_from_spans
                parser = benepar.Parser("benepar_en3")
                tree_vocab = parser._parser.config["label_vocab"]
                tree_vocab_rev = {v: k for k, v in tree_vocab.items()}

                ###############
                control_label_lst = []
                with open('diffusion_lm/improved-diffusion/control_gen/target_spans.json',
                          'r') as controlf:
                    for line in controlf:
                        control_label_lst.append(json.loads(line))
                # print(control_label_lst[:1])
                control_constraints = []
                for label_class_dict in control_label_lst:
                    spans = label_class_dict['spans']
                    spans = [(a+1, b+1, c) for (a,b,c) in spans]
                    assert len(spans) == 1
                    padded_charts = padded_chart_from_spans(tree_vocab, spans)
                    padded_charts = th.LongTensor(padded_charts).unsqueeze(0)
                    print(padded_charts.shape, 'built from spans. ')
                    label_ids = padded_charts
                    langevin_fn_selected = partial(langevin_fn_tree, 0.1, model_control, model3.cuda(),
                                                   label_ids.expand(args.batch_size, -1, -1),
                                                   0.1)
                    print((str(label_class_dict['spans'][0]),))
                    control_constraints.append((langevin_fn_selected, (str(label_class_dict['spans'][0]),)
                                                ))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)


            elif args.eval_task_ == 'control_length':
                control_length_lst = list(range(10, 41)) #[40] #[10, 20, 30]
                control_constraints = []
                for target_length in control_length_lst:
                    encoded_partial_seq = [th.LongTensor([0])]
                    print(encoded_partial_seq)
                    assert len(encoded_partial_seq) == 1
                    right_length = args.image_size ** 2 - len(encoded_partial_seq[0])
                    # right_length = args.tgt_len - len(encoded_partial_seq[0])
                    # assert args.tgt_len > len(encoded_partial_seq[0])
                    right_pad = th.empty(right_length).fill_(todo_pad_token).long()
                    print(right_pad, right_length, len(encoded_partial_seq[0]))
                    encoded_partial_seq = [th.cat([seq, right_pad], dim=0) for seq in encoded_partial_seq]
                    encoded_partial_seq[0][target_length - 1] = tokens2id['END']
                    # encoded_partial_seq[0][target_length] = tokens2id['START']

                    print(encoded_partial_seq[0], todo_pad_token)
                    partial_mask = (encoded_partial_seq[0] == todo_pad_token).unsqueeze(0).expand(args.batch_size, -1)
                    # print(partial_mask[0])
                    # 10/0
                    label = encoded_partial_seq[0]
                    label_ids = th.tensor(label).unsqueeze(0)
                    label_ids = label_ids.masked_fill(label_ids == todo_pad_token, 3)
                    tgt_embs = model3.cuda()(label_ids.cuda())
                    langevin_fn_selected = partial(langevin_fn_length, 0.01, diffusion, partial_mask, model,
                                                   tgt_embs.expand(args.batch_size, -1, -1), 0.1)
                    control_constraints.append((langevin_fn_selected, (str(target_length),)))

                partial_seq = control_constraints
                # print(control_constraints)
                encoded_partial_seq = [encoded_partial_seq[0] for _ in range(len(partial_seq))]
                assert len(partial_seq) == len(encoded_partial_seq)
                print(f'RUNNING FOR {len(partial_seq)} constraints.', '*-' * 20)


        elif args.eval_task_ == 'interpolate':
            print(encoded_partial_seq)
            assert len(encoded_partial_seq[0]) ==  len(encoded_partial_seq[1])
            assert len(encoded_partial_seq[0]) > 1
        print(encoded_partial_seq[0], len(encoded_partial_seq[0]))
    # else: text, using huggingface tokenizer.



    logger.log("sampling...")
    sample_dict = {}
    # model3 = get_weights(model_embs, args)
    if True:
        for (encoded_seq, control_helper) in zip(encoded_partial_seq, partial_seq) :
            all_images = []
            all_labels = []
            print(args.num_samples, encoded_seq.shape, 'encoded_seq.shape')
            while len(all_images) * args.batch_size < args.num_samples:
                model_kwargs = {}
                print(encoded_seq.shape)
                encoded_seq = encoded_seq.unsqueeze(0).expand(args.batch_size,-1)
                print(model_embs.weight.device, encoded_seq.device)
                partial_mask_temp = (encoded_seq == todo_pad_token).view(args.batch_size, -1)
                # encoded_seq[encoded_seq == todo_pad_token] = 0
                encoded_seq.masked_fill_(encoded_seq == todo_pad_token, 3)

                encoded_seq_hidden = model_embs(encoded_seq.cuda())
                seqlen = encoded_seq.size(1)
                if args.model_arch == '1d-unet':
                    encoded_seq_hidden = encoded_seq_hidden.permute(0, 2, 1)
                    partial_mask = partial_mask_temp.unsqueeze(1).expand(-1, args.in_channel, -1)
                    sample_shape = (args.batch_size, args.in_channel, seqlen)
                else:
                    partial_mask = partial_mask_temp.unsqueeze(-1).expand(-1, -1, args.in_channel)
                    sample_shape = (args.batch_size, seqlen, args.in_channel, )
                # print(partial_mask, encoded_seq_hidden.shape)

                if args.eval_task_.startswith('control'):
                    langevin_fn_selected, label_class_attributes = control_helper
                    print('-*'*200, label_class_attributes, '-*'*200)
                    # loop_func_ = diffusion.p_sample_loop_langevin_progressive
                    if args.use_ddim:
                        loop_func_ = diffusion.ddim_sample_loop_progressive
                    else:
                        loop_func_ = diffusion.p_sample_loop_progressive

                    for sample in loop_func_(
                            model,
                            sample_shape,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                            #                     label_ids.expand(args.batch_size, -1), 0.1),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            langevin_fn=langevin_fn_selected,
                            eta=args.eta,
                            # langevin_func=partial(langevin_func, model_control,
                            #                       label_ids.expand(args.batch_size, -1), 0.01),
                    ):
                        final = sample["sample"]


                    if args.verbose == 'yes':
                        with open(f'debug_lst_lgv_{args.notes}.json', 'w') as f:
                            json.dump(debug_lst, f)
                        if  args.eval_task_ == 'control_tree':
                            label_ids = label_ids.expand(args.batch_size, -1, -1).cuda()
                            tgt_embs = model3(label_ids[:, final.size(1):])
                        else:
                            label_ids = label_ids.expand(args.batch_size, -1).cuda()
                            tgt_embs = model3(label_ids[:, final.size(1):])

                        if args.eval_task_ == 'control_attributes':
                            label_ids2 = label_ids.clone()
                            label_ids2[:, :65] = -100
                            # print(label_ids2[:, 65:])
                            # print(final.shape, tgt_embs.shape)
                            input_embs = th.cat([final, tgt_embs], dim=1)
                            model_out = model_control(input_embs=input_embs,
                                                      labels=label_ids2)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits[:, :-1].contiguous()
                            shifted_labels = label_ids2[:, 1:].contiguous()
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)
                        elif args.eval_task_ == 'control_pos':
                            model_out = model_control(input_embs=final,
                                                      pos_ids=label_ids)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits
                            shifted_labels = label_ids
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)
                        elif args.eval_task_ == 'control_tree':
                            model_out = model_control(input_embs=final,
                                                      parse_chart=label_ids)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits
                            shifted_labels = label_ids
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss, loss.shape)
                            # print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).sum(dim=-1).tolist()):
                                print([ww], ll)

                            print(parse_lst[0])
                        else:
                            label_ids2 = th.cat([label_ids[:, :final.size(1)], label_ids], dim=1)
                            label_ids2[:, :64 * 2 + 1] = -100
                            tt = th.LongTensor([0]).expand(final.size(0)).to(final.device)
                            prev_sample = diffusion.q_sample(final, tt)
                            input_embs = th.cat([final, prev_sample, tgt_embs], dim=1)
                            model_out = model_control(input_embs=input_embs,
                                                      labels=label_ids2)
                            print(model_out.loss, 'final end')
                            loss_fn = th.nn.CrossEntropyLoss(reduction='none')
                            shifted_logits = model_out.logits[:, :-1].contiguous()
                            shifted_labels = label_ids2[:, 1:].contiguous()
                            loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)),
                                           shifted_labels.view(-1)).reshape(shifted_labels.shape)
                            print(loss.sum(dim=-1).tolist())
                            word_lst = rounding_func(args.experiment, final, model3, tokenizer)
                            print(len(word_lst))
                            for ww, ll in zip(word_lst, loss.sum(dim=-1).tolist()):
                                print([ww], ll)




                else:
                    label_class_attributes = control_helper
                    loop_func_ = diffusion.p_sample_loop_progressive_infill


                    for sample in loop_func_(
                            model,
                            sample_shape,
                            encoded_seq_hidden,
                            partial_mask,
                            denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                            clip_denoised=args.clip_denoised,
                            model_kwargs=model_kwargs,
                            device=encoded_seq_hidden.device,
                            greedy=False,
                    ):
                        final = sample["sample"]
    
                sample = final
    
    
                if args.model_arch == '1d-unet':
                    print(sample.shape)
                    sample = sample.permute(0, 2, 1)
                    print(sample.shape)
    
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logger.log(f"created {len(all_images) * args.batch_size} samples")
    
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            if args.verbose == 'pipe':
                sample_dict[tuple(label_class_attributes)] = arr
                print(f'writing to sample_dict, for class {" ".join(label_class_attributes)}')

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'

    dist.barrier()
    logger.log("sampling complete")

    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():
            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict

    if args.verbose == 'pipe':
        print(f'sampled for {len(sample_dict)} control tasks')
        out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.json")
        fout = open(out_path_pipe, 'w')
        result_dict = decode_helper(args, sample_dict, diff_model=model)
        for k, word_lst in result_dict.items():
            print({k:word_lst}, file=fout)
        fout.close()
        print(f'written the decoded output to {out_path_pipe}')
        out_path2 = out_path_pipe


    elif args.verbose == 'yes':

        if diffusion.training_mode.startswith('e2e'):
            word_lst_e2e = []
            print('decoding for e2e', )
            print(sample.shape)
            x_t = sample
            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            for seq in cands.indices:
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                word_lst_e2e.append(tokens)
            word_lst = word_lst_e2e
        else:
            logger.log('decode by rounding. ')
            print('load_models')
            set_seed(101)
            print(os.path.split(args.model_path)[0])
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel, os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.notes}.txt")
        fout = open(out_path2, 'w')
        for (xx) in zip( word_lst):
            print(xx[0], file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{shape_str}_{args.notes}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')


    args.out_path2 = out_path2
    return args

def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
#
# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         num_samples=50,#10000,
#         batch_size=16,
#         use_ddim=False,
#         model_path="",
#         model_arch='conv-unet',
#         verbose='yes',
#         out_dir="diffusion_lm/improved_diffusion/out_gen",
#         partial_seq=""
#     )
#     text_defaults = dict(modality='text',
#                          dataset_name='wikitext',
#                          dataset_config_name='wikitext-2-raw-v1',
#                          model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
#                          experiment='gpt2_pre_compress', model_arch='trans-unet',
#                          preprocessing_num_workers=1)
#     defaults.update(model_and_diffusion_defaults())
#     defaults.update(text_defaults)
#     # defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser

def eval(args):
    if args.modality == 'e2e-tgt':
        model_name_path = "predictability/diff_models/e2e-tgt_e=15_b=20_m=gpt2_wikitext-103-raw-v1_101_None"

        COMMAND = f"python scripts/ppl_under_ar.py " \
              f"--model_path {args.model_path} " \
              f"--modality {args.modality}  --experiment random " \
              f"--model_name_or_path {model_name_path} " \
              f"--input_text {args.out_path2}  --mode eval"
        print(COMMAND)
        os.system(COMMAND)


if __name__ == "__main__":
    args = main()
    import numpy as np
    if args.verbose != 'pipe':
        eval(args)

