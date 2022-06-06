"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os, json
import torch as th
import numpy as np
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text, load_synthetic_data
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from functools import partial
from transformers import set_seed
from improved_diffusion.test_util import get_weights, denoised_fn_round, compute_logp, load_results

def main():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)

    training_args['batch_size'] = args.batch_size
    print(args.data_dir)
    del training_args['data_dir']
    # print(args.__dict__, training_args)
    args.__dict__.update(training_args)
    print(args.__dict__['batch_size'], training_args['batch_size'], args.clip_denoised, args.batch_size)
    print(args.data_dir)
    # if args.noise_level > 0.0: flag_noise=True #DEBUG
    args.noise_level = 0.0
    args.roc_train = 'diffusion_lm/ROCstory'
    if args.modality == 'roc-aug':
        args.modality = 'roc'
    # DEBUG
    args.sigma_small = True
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    # diffusion.rescale_timesteps = False # IMPORTANT DEBUG -->  REMOVE
    model.to(dist_util.dev())
    model.eval() # DEBUG

    logger.log("creating data loader...")
    if args.modality == 'image':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
        )
    elif args.modality == 'permuted_image':
        # perm = np.arange(args.image_size * args.image_size)
        # np.random.shuffle(perm)
        model_path_base = os.path.split(args.model_path)[0]
        print(f'load permutation to {model_path_base}/permutation.json')
        with open(f'{model_path_base}/permutation.json', 'r') as f:
            perm = json.load(f)
        perm = np.array(perm)
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            permutation=perm
        )
    elif args.modality == 'synth':
        from improved_diffusion.rounding import load_models
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                    os.path.split(args.model_path)[0])

        data = load_synthetic_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            split='train',
            # split='valid',
            deterministic=True

        )
    elif args.modality == 'pos':
        from improved_diffusion.rounding import load_models
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        data = load_synthetic_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            pos=True,
            deterministic = True
        )
    else:
        from improved_diffusion.rounding import load_models
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        # print(tokenizer)
        # rev_tokenizer = {k:int(v)  for k, v in tokenizer.items()}
        rev_tokenizer = {v:k  for k, v in tokenizer.items()}

        if args.training_mode == 'e2e':
            print('e2e, load the right model embeddings', '*'*80)
            model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

        # print(rev_tokenizer)
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split=args.split,
            load_vocab=rev_tokenizer,
        )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised, args, model2)




def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised, args, model2):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    model3 = get_weights(model2, args)
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        model_kwargs['mapping_func'] = partial(compute_logp, args, model3.cuda())
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            # denoised_fn=None,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0) / dist.get_world_size()
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        total_bpd = minibatch_metrics["total_bpd"]
        total_bpd = total_bpd.mean() / dist.get_world_size()
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())
        num_complete += dist.get_world_size() * batch.shape[0]

        logger.log(f"done {num_complete} samples on {args.split}: bpd={np.mean(all_bpd)}, "
                   f"per token={np.mean(all_bpd) * args.in_channel} ", args.model_path)
        temp_cat = np.mean(np.stack(all_metrics['vb']), axis=0)
        if len(temp_cat) % 8 == 0:
            print([y.sum() for y in np.split(np.mean(np.stack(all_metrics['vb']), axis=0), 8)])
        else:
            print(temp_cat[0].sum())
            print([y.sum() for y in np.split(temp_cat[1:-1], 8)])
            print(temp_cat[-1].sum())
        vb_temp = np.mean(np.stack(all_metrics['vb']), axis=0)
        print(vb_temp.shape, vb_temp.sum())
        print(vb_temp[-10:])


    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            model_base_name = os.path.basename(
                os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
            # args.out_dir = os.path.join(args.out_dir, f"{model_base_name}.samples_{shape_str}.txt")
            out_path = os.path.join(args.out_dir, f"{model_base_name}.{name}_{args.split}_{args.clamp}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.mean(np.stack(terms), axis=0))

    dist.barrier()
    logger.log("evaluation complete")

    if 'ema' in args.model_path:
        json_path = os.path.join(os.path.split(args.model_path)[0], f'ema_score_{args.split}_nll.json')
    elif args.clamp == 'noclamp':
        json_path = os.path.join(os.path.split(args.model_path)[0], f'score_{args.split}_nll_noclamp.json')
    else:
        json_path = os.path.join(os.path.split(args.model_path)[0], f'score_{args.split}_nll.json')

    print(f'written to {json_path}')
    temp_cat = np.mean(np.stack(all_metrics['vb']), axis=0)
    if len(temp_cat) % 8 == 0:
        temp_cat = temp_cat
    else:
        temp_cat = temp_cat[1:-1]
    json_dict = {
        f'score_{args.split}_ppl_token': np.mean(all_bpd) * args.in_channel,
        f'score_{args.split}_ppl_dim': np.mean(all_bpd),
        f'break_down_{args.split}_dim' : [y.sum().item() for y in np.split(temp_cat, 8)],
        f'last_10_{args.split}_dim': vb_temp[-10:].tolist(),
        'source_file': out_path,
        'num_samples':num_samples,
    }
    load_results(json_path, json_dict)


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, num_samples=128, batch_size=64, model_path="",
        out_dir="diffusion_lm/improved_diffusion/scores",
        emb_scale_factor=1.0, split='train', debug_path='', clamp='clamp',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
