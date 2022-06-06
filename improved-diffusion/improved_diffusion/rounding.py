import torch
# bert results
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
import sys, yaml, os
# print( os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# sys.path.insert(0, 'diffusion_lm/transformers/examples/pytorch/language-modeling')
# sys.path.insert(0, os.path.join(sys.path[0], '../../transformers/examples/pytorch/language-modeling'))
# from custom_trainer import GPT2LMHeadModelCompress, BERTModelCompress, AutoEncoderWithNoise

def load_models(modality, mode, model_name_or_path, emb_dim, file, extra_args=None):

    if mode in ['random', 'random1', 'random_up_proj', 'glove']:
        if modality == 'synth':
            print(file, 'deciding what to load::: ')
            if 'synth128' in file:
                config = 'diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml'
            else:
                config = 'diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml'
            import sys, os
            sys.path.insert(0, 'diffusion_lm/synthetic_data/rnns-stacks')
            from dataset import Dataset as SynthDataset
            args_synth = yaml.load(open(config))
            dataset = SynthDataset(args_synth)
            model = torch.nn.Embedding(len(dataset.vocab), emb_dim)
            print('initializing the random embeddings', model)
            # print(os.path.split(file.split('.')[0])[-1])
            # path_save = '{}/random_emb.torch'.format(file)
            path_save = '{}/random_emb.torch'.format(file)
            model.load_state_dict(torch.load(path_save))
            print(dataset.vocab)
            tokenizer = {v: k for k, v in dataset.vocab.items()}
        else:
            import json
            if modality == 'book' or (extra_args is not None and extra_args.use_bert_tokenizer == 'yes'):
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                if 'e2e' in file and modality == 'book':
                    emb_dim = 1
            else:
                path_save_tokenizer = '{}/vocab.json'.format(file)
                print(f'loading from {path_save_tokenizer}')
                with open(path_save_tokenizer, 'r') as f:
                    vocab = json.load(f)
                print(len(vocab))
                tokenizer = {v: k for k, v in vocab.items()}
            model = torch.nn.Embedding(len(tokenizer), emb_dim)
            path_save = '{}/random_emb.torch'.format(file)
            model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(modality, mode, model_name_or_path):
    if mode in ['random', 'random_up_proj', 'glove']:
        if modality == 'synth':
            print(model_name_or_path, 'deciding what to load::: ')
            if 'synth128' in model_name_or_path:
                config = 'diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml'
            else:
                config = 'diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k32_trainc20000.yaml'

            import sys, os
            sys.path.insert(0, 'diffusion_lm/synthetic_data/rnns-stacks')
            from dataset import Dataset as SynthDataset
            args_synth = yaml.load(open(config))
            dataset = SynthDataset(args_synth)
            tokenizer = {v: k for k, v in dataset.vocab.items()}
        elif modality =='book':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            import json
            path_save_tokenizer = '{}/vocab.json'.format(model_name_or_path)
            with open(path_save_tokenizer, 'r') as f:
                vocab = json.load(f)
            tokenizer = {v: k for k, v in vocab.items()}

    return tokenizer

def rounding_func(mode, text_emb_lst, model, tokenizer, emb_scale_factor=1.0):
    decoded_out_lst = []
    if mode in ['random', 'random_up_proj', 'glove']:
        down_proj_emb = model.weight  # input_embs
        down_proj_emb2 = None


        def get_knn(down_proj_emb, text_emb, dist='cos'):

            if dist == 'cos':
                adjacency = down_proj_emb @ text_emb.transpose(1, 0).to(down_proj_emb.device)
            elif dist == 'l2':
                adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                    down_proj_emb.size(0), -1, -1)
                adjacency = -torch.norm(adjacency, dim=-1)
            topk_out = torch.topk(adjacency, k=6, dim=0)
            return topk_out.values, topk_out.indices

        dist = 'l2'
        # print(npzfile['arr_0'].shape)
        for text_emb in text_emb_lst:
            import torch
            text_emb = torch.tensor(text_emb)
            # print(text_emb.shape)
            if len(text_emb.shape) > 2:
                text_emb = text_emb.view(-1, text_emb.size(-1))
            else:
                text_emb = text_emb
            val, indices = get_knn((down_proj_emb2 if dist == 'cos' else down_proj_emb),
                                   text_emb.to(down_proj_emb.device), dist=dist)
            # generated_lst.append(tuple(indices[0].tolist()))

            # print(indices[0].tolist())
            # for i in range(64):
            #     print([tokenizer[x.item()] for x in indices[:,i]])
            decoded_out = " ".join([tokenizer[i] for i in indices[0].tolist()])
            decoded_out_lst.append(decoded_out)

    return decoded_out_lst

