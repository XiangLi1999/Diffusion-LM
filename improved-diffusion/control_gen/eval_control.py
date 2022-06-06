import torch, argparse, json
import benepar, spacy_stanza
import numpy as np
import sys, os
import csv
from nltk.tree import Tree
sys.path.insert(0, os.path.join(sys.path[0], '../scripts/'))
from tree_helper import chart_from_tree, pad_charts, padded_chart_from_spans
sys.path.insert(0, os.path.join(sys.path[0], '../../misc/self-attentive-parser/src/'))
import evaluate
from spacy.lang.en import English
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
nlp = English()
tokenizer_spacy = nlp.tokenizer

def eval_ppl2(args, text_samples):
    print(f'loading from {args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # path to the AR model trained for LMing this task.
    ).cuda()

    if 'r2l' in args.model_name_or_path:
        print('Use the right-to-left encoding.')

    args.model_path = 'predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_' \
                      'lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart/ema_0.9999_200000.pt'
    tokenizer = load_tokenizer('e2e-tgt', 'random', os.path.split(args.model_path)[0])
    # print(args.modality, tokenizer, args.experiment)
    reverse_tokenizer = {v: k for k, v in tokenizer.items()}
    full_score = []
    for idxx, (gold, full_word_lst) in enumerate(text_samples.items()):
        # print(len(full_word_lst), full_word_lst[0])
        agg_loss = []
        for x in full_word_lst:
            # x = " ".join(x).split()
            if 'r2l' in args.model_name_or_path:
                string = ["START"] + list(reversed(x)) + ["END"]
                tokenized_x = [reverse_tokenizer.get(s, reverse_tokenizer['UNK']) for s in string]
            else:
                tokenized_x = [reverse_tokenizer['START']] + [reverse_tokenizer.get(s, reverse_tokenizer['UNK']) for s in x] \
                              + [reverse_tokenizer['END']]
            # print(tokenized_x)
            tokenized_x = torch.LongTensor(tokenized_x).cuda()
            labels = tokenized_x.clone()
            labels[labels == reverse_tokenizer['PAD']] = -100
            model_output = model(tokenized_x, labels=labels)
            # print(model_output.loss)
            # if idxx == 3:
            #     print(tokenized_x, model_output.loss.item())
            agg_loss.append(model_output.loss.item())
        example_mean_score = torch.tensor(agg_loss).mean()
        # print(f'\nthe mean loss is {example_mean_score} for index', idxx )
        full_score.append(example_mean_score)
    full_score_ = np.array(full_score).mean()
    print(f'full NLL score is {full_score_} for {len(full_score)}')
    print(f'full PPL score is {np.e ** full_score_} for {len(full_score)}')



def eval_ppl(args, text_samples):
    '''
    Evaluating using GPT2 finetuned on this task...
    :param text_lst:
    :return:
    '''

    # load model
    print(f'loading from {args.model_name_or_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,  # path to the AR model trained for LMing this task.
    ).cuda()

    # load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print('finished loading models.')

    args.model_path = 'predictability/diffusion_models_v6/diff_e2e-tgt_pad_rand16_transformer_' \
                      'lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart/ema_0.9999_200000.pt'
    diff_tokenizer = load_tokenizer('e2e-tgt', 'random', os.path.split(args.model_path)[0])
    reverse_diff_tokenizer = {v: k for k, v in diff_tokenizer.items()}

    full_score = []
    for gold, full_word_lst in text_samples.items():
        agg_loss = []
        for x in full_word_lst:
            x = [kk if kk in reverse_diff_tokenizer else 'UNK' for kk in x]
            x = tokenizer.bos_token + " ".join(x) + tokenizer.eos_token
            # print(x)
            # should also add BOS EOS token?

            tokenized_x = tokenizer(x, return_tensors='pt') #[reverse_tokenizer[s] for s in x]
            input_ids = tokenized_x['input_ids'].cuda()
            labels = input_ids.clone()
            # print(tokenized_x)
            # tokenized_x = torch.LongTensor(tokenized_x).cuda()
            # labels = tokenized_x.clone()
            # labels[labels == reverse_tokenizer['PAD']] = -100
            model_output = model(input_ids, labels=labels)
            agg_loss.append(model_output.loss.item())
        example_mean_score = torch.tensor(agg_loss).mean()
        # print(f'\nthe mean loss is {example_mean_score}', )
        full_score.append(example_mean_score)
    full_score_ = np.array(full_score).mean()
    print(f'full NLL score is {full_score_} for {len(full_score)}')
    print(f'full PPL score is {np.e ** full_score_} for {len(full_score)}')


def read_files(args):
    '''
    :param args:
    :return: list of tokenized sentences.
    '''
    if args.input_format == 'file':
        text_samples = []
        if args.input_text.endswith('json'):
            with open(args.input_text, 'r') as f:
                for line in f:
                    words = [x.text for x in tokenizer_spacy(json.loads(line)[0])]
                    text_samples.append(words)
                    # text_samples.append(json.loads(line)[0].split(' '))


        else:
            with open(args.input_text, 'r') as f:
                for line in f:
                    text_samples.append(line.strip().split())

        # remove trailing PAD tokens.
        text_samples2 = []
        for sent in text_samples:
            tempsent = [x for x in sent if x != 'PAD']
            if tempsent[0] == 'START':
                tempsent = tempsent[1:]
            if tempsent[-1] == 'END':
                tempsent = tempsent[:-1]
            if tempsent[-1] == '\n' and args.mode in ['e2e-tgt-tree', 'e2e-tgt-tree-paired']:
                tempsent[-1] = '.'
            text_samples2.append(tempsent)
        return text_samples2
    elif args.input_format == 'paired':
        import ast
        # nlp = English()
        # tokenizer = nlp.tokenizer
        result_lst = defaultdict(list)

        if args.input_text.endswith('json'):
            with open(args.input_text, 'r') as f:
                for line in f:
                    try:
                        line = json.loads(line)
                    except:
                        if args.mode == 'e2e-tgt-spans-paired':
                            line = ast.literal_eval(line)
                            line = {tuple(ast.literal_eval(k[0])) : v for k, v in line.items()}
                            result_lst.update(line)
                        else:
                            line = ast.literal_eval(line)
                            result_lst.update(line)

        elif args.input_text.endswith('log'):
            with open(args.input_text, 'r') as csvfile:
                roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
                for idx, row in enumerate(roc_reader):
                    if idx == 0: continue
                    if args.mode == 'e2e-tgt-spans-paired' or args.mode == 'e2e-tgt-length-paired':
                        pos = tuple(ast.literal_eval(row[0]))

                        if args.mode == 'e2e-tgt-length-paired':
                            pos = list(pos)
                            pos[0] = int(pos[0]) + 2 # because this count didn't accounted for START and END
                            pos = tuple(pos)
                    else:
                        pos = tuple(row[0].split())
                    result_lst[pos].append(row[2])

        clean_result_lst = {}
        for k, text_samples in result_lst.items():
            text_samples2 = []
            for sent in text_samples:
                sent = sent.split(' ')
                # KEY DEBUG.
                # sent = [x.text for x in tokenizer_spacy(sent)]
                # print(sent, sent2)
                # 10/0
                tempsent = [x for x in sent if x != 'PAD']
                if tempsent[0] == 'START':
                    tempsent = tempsent[1:]
                if tempsent[-1] == 'END':
                    tempsent = tempsent[:-1]
                if tempsent[-1] == '\n' and args.mode == 'e2e-tgt-tree':
                    tempsent[-1] = '.'

                # KEY DEBUG.
                tempsent = " ".join(tempsent)
                tempsent = [x.text for x in tokenizer_spacy(tempsent)]
                text_samples2.append(tempsent)
            if k[0] == 'START' and k[-1] == 'END':
                kk_ = k[1:-1]
            else:
                kk_ = k
            clean_result_lst[kk_] = text_samples2 # remove start and end from the training data.
        return clean_result_lst

def eval_parse(parser, generated, tree_vocab):
    sent_lst = []
    for sent in generated:
        # print(sent)
        input_sentence1 = benepar.InputSentence(
            words=sent,
        )
        sent_lst.append(input_sentence1)
    parse_lst = list(parser.parse_sents(sent_lst))
    # print(examples['text'][:10])
    assert len(parse_lst) == len(generated)
    # print(parse_lst[:2])
    spans_lst = []
    for parse in parse_lst:
        chart, spans = chart_from_tree(tree_vocab, parse, verbose=True)
        spans_lst.append(spans)
    return parse_lst, spans_lst

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def score_spans(gold_spans, generated_span):
    print(gold_spans)
    print(generated_span)
    gold_spans = set([gold_spans])
    generated_span = set(generated_span)
    intersection = gold_spans.intersection(generated_span)
    print(intersection, len(intersection) / len(gold_spans))
    # union = gold_spans.union(generated_span)
    # print(len(union), len(intersection))

    # if unlabeled:
    # print(generated_span)
    # unlabeled_gold_spans = set([(a,b) for (a, b, v) in gold_spans])
    # unlabeled_generated_span =set([(a,b) for (a, b, v) in generated_span])
    # intersection = gold_spans.intersection(generated_span)
    # union = gold_spans.union(generated_span)
    return len(intersection) / len(gold_spans)

def score_tree(gold_tree, pred_trees):
    # print([x.leaves() for x in pred_trees])

    def reset_leaves(tree_):
        simple_increm = 0
        for s in tree_.subtrees(lambda t: t.height() == 2):
            s[0] = simple_increm
            s._label = 'NN'
            simple_increm += 1
        return simple_increm

    # reset.
    increm_gold = reset_leaves(gold_tree)
    # print(increm_gold)
    for i, pred in enumerate(pred_trees):
        increm_pred = reset_leaves(pred)
        # print(increm_pred, 'pred', i)

    use_evalb = True
    if use_evalb:
        # print(len(gold_tree), len(pred_trees), gold_tree)
        gold_trees = [gold_tree] * len(pred_trees)
        print(len(gold_tree.leaves()), [len(x.leaves()) for x in pred_trees])
        # print(pred_trees[0])
        dev_fscore = evaluate.evalb('diffusion_lm/misc/self-attentive-parser/EVALB',
                                    gold_trees, pred_trees)
        print(dev_fscore)

    return dev_fscore

def score_pos(gold_pos, generated_pos):
    ed = levenshteinDistance(gold_pos, generated_pos)
    return 1 - (ed / len(gold_pos))

def score_pos_em(gold_pos, generated_pos):
    # print(len(gold_pos), len(generated_pos), gold_pos, generated_pos)
    if len(generated_pos) > len(gold_pos):
        generated_pos = generated_pos[:len(gold_pos)]
    elif len(generated_pos) < len(gold_pos):
        generated_pos = generated_pos + ['PAD'] * (len(gold_pos) - len(generated_pos))
    assert len(gold_pos) == len(generated_pos)
    correct = 0
    all = 0
    for x1, x2 in zip(gold_pos, generated_pos):
        if x1 == x2:
            correct += 1
        all += 1
    return correct/all

def score_attributes(gold_att, generated):
    if gold_att in generated:
        return 1.
    else:
        return 0.

def eval_pos(tagger, generated_text):
    generated_pos = []
    for sent in generated_text:
        sent_full = " ".join(sent)
        doc = tagger(sent_full)
        generated_pos.append([token.pos_ for token in doc])
    return generated_pos

def eval_(args, text_samples):
    if args.mode == 'e2e-tgt-tree':

        parser = benepar.Parser("benepar_en3")
        tree_vocab = parser._parser.config["label_vocab"]
        if args.gold_ref == 'full':
            # toy1 = 'START Located in riverside area , Alimentum restaurant is a place to bring the whole family . \n END'.split()
            # toy1 = 'START Alimentum is not a family - friendly place , located in city centre . \n END'.split()
            toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating',
                    '.', 'Prices', 'start', 'at', '£', '30', '.', 'END']
            input_sentence1 = benepar.InputSentence(
                words=toy1[1:-1],
            )
            gold_parse = list(parser.parse_sents([input_sentence1]))[0]
            chart, gold_spans = chart_from_tree(tree_vocab, gold_parse, verbose=True)
            print(len(toy1[1:-1]), len(list(gold_parse.leaves())))
        elif args.gold_ref == 'span':
            # spans = [(10, 14, 'ADJP')]
            gold_spans = [(0, 4, 'S::VP')]
            gold_spans = [(0, 0, 'NP')]
            gold_spans = [(9, 13, 'ADJP')]
            # gold_spans = [(9, 13, 'PP')]

        print(text_samples[:1])
        # correct for length:
        target_len = len(gold_parse.leaves())
        print(gold_parse.leaves(), 'target')
        for i, x in enumerate(text_samples):
            if len(x) == target_len:
                continue
            elif len(x) > target_len:
                text_samples[i] = x[:target_len]
            else:
                print('padded to same length', (target_len-len(x)))
                text_samples[i] = x + ['.'] * (target_len-len(x))
                # print(text_samples[i])
                # print('SAD, our model is shorter??')
        generated_parse, generated_span = eval_parse(parser, text_samples, tree_vocab)
        # print(gold_spans)
        # print(generated_span[:2])
        evalb_score = score_tree(gold_parse, generated_parse)
        print([len(x) for x in text_samples])
        score_lst = []
        for x in generated_span:
            score_lst.append(score_spans(gold_spans, x))

        print(np.array(score_lst).mean())
    elif args.mode == 'e2e-tgt-pos':
        tagger = spacy_stanza.load_pipeline("en", processors='tokenize,mwt,pos', ) #processors={"tokenize": "spacy",}
        if args.gold_ref == 'full':
            toy1 = 'START The Mill is a coffee shop with an expensive menu near The Sorrento . \n END'.split()
            toy1 = ['START', 'The', 'Vaults', 'pub', 'near', 'Café', 'Adriatic', 'has', 'a', '5', 'star', 'rating', '.',
                    'Prices', 'start', 'at', '£', '30', '.', '\n', 'END']
            sent_full = " ".join(toy1[1:-1])
            doc = tagger(sent_full)
            gold_pos = [token.pos_ for token in doc]
        elif args.gold_ref == 'span':
            gold_pos = [(9, 'PROPN')]

        generated_pos = eval_pos(tagger, text_samples)
        score_lst = []
        score_lst2 = []
        for x in generated_pos:
            print(gold_pos)
            print(x)
            print()
            score_lst.append(score_pos(gold_pos, x))
            score_lst2.append(score_pos_em(gold_pos, x))

        print(np.array(score_lst).mean())
        print(np.array(score_lst2).mean())
    elif args.mode == 'e2e-tgt-pos-paired':
        import stanza
        nlp = spacy_stanza.load_pipeline("en", processors={"tokenize": "spacy"})
        print(nlp)
        # nlp = stanza.Pipeline("en", processors={"tokenize": "spacy", 'pos': 'combined'}, package=None)

        full_score = []
        for gold, full_word_lst in text_samples.items():
            print(gold, len(full_word_lst), full_word_lst[:2])
            # full_word_lst = full_word_lst[:2]
            sent_lst = [" ".join(seq) for seq in full_word_lst]
            sent_full = " ".join(sent_lst)
            # print(sent_lst)
            try:
                doc = nlp(sent_full)
                doc_token_pos = [(token.text, token.pos_,) for token in doc]
                len_lst = [len(seq) for seq in full_word_lst]
                print(sum(len_lst), len(doc_token_pos), 'should be equal!!! ')
                assert sum(len_lst) == len(doc_token_pos)
                pos_lst = []
                init_idx = 0
                for len_temp in len_lst:
                    pos_lst.append([x[1] for x in doc_token_pos[init_idx:init_idx + len_temp]])
                    init_idx = init_idx + len_temp

            except:
                print(f'stanza pipeline failed... for this {gold}')

                # parse each sentence separately...
                pos_lst = []
                for single_sent in sent_lst:
                    doc = nlp(single_sent)
                    # doc_token_pos = [(token.text, token.pos_,) for token in doc]
                    pos_lst.append([ token.pos_ for token in doc])


            score_lst = []
            score_lst2 = []
            for x in pos_lst:
                score_lst.append(score_pos(gold, x))
                score_lst2.append(score_pos_em(gold, x))
            score_ed = np.array(score_lst).mean()
            score_em = np.array(score_lst2).mean()
            print(len(score_lst), score_ed, score_em)
            full_score.append(score_em)
        full_score_em = np.array(full_score).mean()
        print(full_score_em, f"\pm {np.array(full_score).std()}", len(full_score))

    if args.mode == 'e2e-tgt-tree-paired':

        parser = benepar.Parser("benepar_en3")
        tree_vocab = parser._parser.config["label_vocab"]

        full_score = []
        for idx, (gold_parse, full_word_lst) in enumerate(text_samples.items()):
            # to avoid evalb complain --> change \n to .
            gold_parse_str = gold_parse[0]
            gold_parse_str = gold_parse_str.replace('\n', '.')
            # print([gold_parse_str], 'gold tree string ')
            gold_parse = Tree.fromstring(gold_parse_str)
            target_len = len(gold_parse.leaves())
            # print(gold_parse.leaves(), 'target')
            # print(full_word_lst)
            for i, x in enumerate(full_word_lst):
                if len(x) == target_len:
                    continue
                elif len(x) > target_len:
                    print('generated seq is longer than gold seq')
                    full_word_lst[i] = x[:target_len]
                else:
                    print('padded to same length', (target_len - len(x)))
                    full_word_lst[i] = x + ['.'] * (target_len - len(x))
                    # print(text_samples[i])
                    # print('SAD, our model is shorter??')
            generated_parse, generated_span = eval_parse(parser, full_word_lst, tree_vocab)
            evalb_score = score_tree(gold_parse, generated_parse) # inputs are nltk.Tree
            # print(type(evalb_score))
            print(evalb_score.fscore)
            full_score.append(evalb_score.fscore)
        full_score_f1 = np.array(full_score).mean()
        # print(full_score_f1, len(full_score))
        print(full_score_f1, f"\pm {np.array(full_score).std()}", len(full_score))

    elif args.mode == 'e2e-tgt-spans-paired':

        parser = benepar.Parser("benepar_en3")
        tree_vocab = parser._parser.config["label_vocab"]

        full_score = []
        for idx, (gold_spans, full_word_lst) in enumerate(text_samples.items()):
            # to avoid evalb complain --> change \n to .
            print(gold_spans, '11 gold')
            generated_parse, generated_span = eval_parse(parser, full_word_lst, tree_vocab)
            score_lst = []
            for x in generated_span:
                score_lst.append(score_spans(gold_spans, x))
            print(score_lst)
            score_lst_mean = np.array(score_lst).mean()
            full_score.append(score_lst_mean)
        full_score_span = np.array(full_score).mean()
        print(full_score_span, f"\pm {np.array(full_score).std()}", len(full_score))

    if args.mode == 'e2e-tgt-attribute-paired':

        full_score = []
        for idx, (attribute, full_word_lst) in enumerate(text_samples.items()):
            # print(attribute)
            attribute = " ".join(attribute).split(':')[1].strip()
            gold_attribute = attribute
            score_lst = []
            for i, x in enumerate(full_word_lst):
                # print(gold_attribute, x)
                score_lst.append(score_attributes(gold_attribute, " ".join(x)))
            score_lst_mean = np.array(score_lst).mean()
            full_score.append(score_lst_mean)
        full_score_mean = np.array(full_score).mean()
        # print(full_score_mean, len(full_score))
        print(full_score_mean, f"\pm {np.array(full_score).std()}", len(full_score))

    if args.mode == 'e2e-tgt-length-paired':

        full_score = []
        for idx, (attribute, full_word_lst) in enumerate(text_samples.items()):
            tgt_len = int(attribute[0]) - 2 # remove START and END.
            score_lst = []
            for i, x in enumerate(full_word_lst):
                if tgt_len == len(x):
                # if np.abs(tgt_len - len(x)) <= 2:
                    score_lst.append(1.)
                else:
                    score_lst.append(0.)
            score_lst_mean = np.array(score_lst).mean()
            full_score.append(score_lst_mean)
        full_score_mean = np.array(full_score).mean()
        # print(full_score_mean, len(full_score))
        print(full_score_mean, f"\pm {np.array(full_score).std()}", len(full_score))

    elif args.mode == 'e2e-tgt-attribute':
        gold_attribute = ""
        score_lst = []
        for x in text_samples:
            score_lst.append(score_attributes(gold_attribute, x))
        print(np.array(score_lst).mean())

if __name__ == '__main__':

    # 'diffusion_lm/improved_diffusion/out_gen/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart.ema_0.9999_200000.pt.infill_control_tree_50x64x16_tree_partial-cat-lgv0.1.json'
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--input_text', type=str, default='diffusion_lm/improved_diffusion/out_gen/diff_e2e-tgt_pad_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_xstart.ema_0.9999_200000.pt.'
                                                          'infill_control_tree_50x64x16_tree_partial-cat-lgv0.1.json',)
    parser.add_argument('--input_format', type=str, default='batch', help='wp, wikitext')

    parser.add_argument('--mode', type=str, default='e2e-tgt-tree', help='')
    parser.add_argument('--gold_ref', type=str, default='full', help='')
    parser.add_argument('--model_name_or_path', type=str, default='predictability/diff_models/e2e-tgt_e=20_b=64_m=gpt2_wikitext-103-raw-v1_101_wp_finetune_UNK', help='')
                        # default='predictability/diff_models/e2e-tgt_e=6_b=10_m=gpt2_wikitext-103-raw-v1_101_wp_pad', help='')
    


    args = parser.parse_args()
    text_samples = read_files(args)
    eval_(args, text_samples)
    eval_ppl(args, text_samples)
    # eval_ppl2(args, text_samples)













