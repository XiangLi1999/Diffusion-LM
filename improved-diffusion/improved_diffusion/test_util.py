import torch as th
import numpy as np

def compute_logp(args, model, x, input_ids):
    word_emb = model.weight
    sigma = 0.1
    if args.model_arch == '1d-unet':
        x = x.permute(0, 2, 1)

    bsz, seqlen, dim = x.shape

    x_flat = x.reshape(-1, x.size(-1)).unsqueeze(0)  # 1, bsz*sample*seqlen, dim
    word_emb_flat = word_emb.unsqueeze(1)  # vocab, 1,  dim
    diff = (x_flat - word_emb_flat) ** 2  # vocab, seqlen, dim

    logp_expanded = -diff.sum(dim=-1) / (2 * sigma ** 2)  # vocab, seqlen
    logp_expanded = logp_expanded.permute((1, 0))
    # print(th.topk(logp_expanded.view(bsz, seqlen, -1), k=5, dim=-1)[0])
    # print(input_ids[0])
    ce = th.nn.CrossEntropyLoss(reduction='none')
    loss = ce(logp_expanded, input_ids.view(-1)).view(bsz, seqlen)
    # print(loss[0])

    # print(loss.shape)
    return loss

def get_weights(model, args):
    if hasattr(model, 'transformer'):
        input_embs = model.transformer.wte  # input_embs
        down_proj = model.down_proj
        down_proj_emb = down_proj(input_embs.weight)
        print(down_proj_emb.shape)
        # model = th.nn.Embedding(down_proj_emb.shape[1], down_proj_emb.shape[0])
        model = th.nn.Embedding(down_proj_emb.size(0), down_proj_emb.size(1))
        print(args.emb_scale_factor)
        model.weight.data = down_proj_emb * args.emb_scale_factor

    elif hasattr(model, 'weight'):
        pass
    else:
        assert NotImplementedError
        
    model.weight.requires_grad = False
    return model

def denoised_fn_round(args, model, text_emb, t):
    # thresh_t = 50
    # # print(thresh_t)
    # if thresh_t is not None and t[0] > thresh_t:
    #     return text_emb

    if args.model_arch == '1d-unet':
        text_emb = text_emb.permute(0, 2, 1)
    # return text_emb
    # print(t.float().mean(), t[0])

    # assert t.float().mean() == t[0].float()
    
    # print(text_emb.shape) # bsz, seqlen, dim
    down_proj_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb**2).sum(-1).view(-1, 1) #vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1) #d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1) #bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(down_proj_emb, text_emb_t) #(vocab, d) x (d, bsz*seqlen)
            dist = th.clamp(dist, 0.0, np.inf)
            # print(dist.shape)
        topk_out = th.topk(-dist, k=1, dim=0)
        #     adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
        #         down_proj_emb.size(0), -1, -1)
        #     adjacency = -th.norm(adjacency, dim=-1)
        # topk_out = th.topk(adjacency, k=1, dim=0)
        # print(topk_out1.indices == topk_out.indices)
        # assert th.all(topk_out1.indices == topk_out.indices)
        return topk_out.values, topk_out.indices

    def get_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                down_proj_emb.size(0), -1, -1)
            adjacency = -th.norm(adjacency, dim=-1)
        topk_out = th.topk(adjacency, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(down_proj_emb,
    #                        text_emb.to(down_proj_emb.device), dist=dist)
    val, indices = get_efficient_knn(down_proj_emb,
                           text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    if args.model_arch == '1d-unet':
        new_embeds = new_embeds.permute(0, 2, 1)
    return new_embeds

def load_results(json_path, load_dict):
    import json
    with open(json_path, 'w') as f:
        json.dump(load_dict, f, indent=2)
