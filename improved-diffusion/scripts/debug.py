"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json

import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion.text_datasets import load_data_text, load_synthetic_data
from improved_diffusion import dist_util, logger
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.test_util import get_weights, denoised_fn_round, compute_logp
from improved_diffusion.gaussian_diffusion import _extract_into_tensor
from transformers import set_seed
from functools import partial

def main():
    set_seed(101)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)
    args.noise_level = 0.0
    args.sigma_small = True
    if args.experiment == 'random1': args.experiment = 'random'

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    if args.modality == 'image':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
        )
    elif args.modality == 'synth':
        from improved_diffusion.rounding import load_models
        # model2, tokenizer = load_models('synth', 'random', None, args.in_channel,
        #                                 os.path.split(args.model_path)[0])

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
            deterministic=True
        )
    else:
        from improved_diffusion.rounding import load_models
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        rev_tokenizer = {v: k for k, v in tokenizer.items()}
        if args.training_mode == 'e2e':
            print('e2e, load the right model embeddings', '*'*80)
            model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
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
            split='valid',
            load_vocab=rev_tokenizer,
        )

    logger.log(f"debugging (starting from step {args.t_start})...")
    all_images = []
    all_labels = []
    ground_truth=[]
    print(args.num_samples)
    model3 = get_weights(model2, args)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        final = None
        def get_noise_from_final(final_gold):
            final_gold = final_gold.cuda()
            print(final_gold.shape)
            batch_size = final_gold.size(0)
            t_batch = th.tensor([args.t_start] * batch_size, device=final_gold.device)
            noise = diffusion.q_sample(final_gold, t_batch)
            return noise

        batch, cond = next(data)
        noise = get_noise_from_final(batch)
        print(noise.shape, noise.device)
        print(cond.keys())
        # naive reconstruct:
        # batch_size = batch.size(0)
        # t_batch = th.tensor([args.t_start] * batch_size, device=batch.device)
        # coeff = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t_batch, batch.shape).to(noise.device)
        # naive_recon = noise / coeff
        # recon_ = denoised_fn_round(args, model3.cuda(), naive_recon, t_batch)
        # mse = ((recon_ - batch.to(recon_.device)) ** 2).mean()
        # print(mse, 'naive_mst')
        # # ppl:
        # nll = compute_logp(args, model3.cuda(), naive_recon, cond['input_ids'].to(recon_.device))
        # print(nll.mean())
        # print(diffusion.sqrt_one_minus_alphas_cumprod[args.t_start])
        #
        # return

        ground_truth.append(cond['input_ids'])

        if args.model_arch == '1d-unet':
            sample_shape = (args.batch_size,  args.in_channel, args.image_size ** 2)
        elif args.model_arch == 'conv-unet':
            sample_shape = (args.batch_size, args.in_channel, args.image_size, args.image_size)
        else:
            sample_shape = (args.batch_size, args.image_size ** 2, args.in_channel)

        for sample in diffusion.p_debug_loop_progressive(
                model,
                sample_shape,
                noise=noise,
                clip_denoised=args.clip_denoised,
                # denoised_fn=partial(denoised_fn_round,args,  model3.cuda()), #EDIT
                model_kwargs=model_kwargs,
                custom_t_start=args.t_start,
        ):
            final = sample["sample"]
        print(final.shape)

        if args.model_arch == '1d-unet':
            # print(sample.shape)
            sample = final.permute(0, 2, 1)
            print(sample.shape)
        elif args.model_arch == 'conv-unet':
            sample = final.permute(0, 2, 3, 1)
            sample = sample.contiguous()
        else:
            sample=final

        if diffusion.training_mode == 'e2e':
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
            sample = cands.indices
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            print(tokenizer)
            for seq in cands.indices:
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                word_lst_e2e.append(tokens)

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
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
        out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{shape_str}.{args.t_start}.npz")
        # out_path = os.path.join(args.out_dir, f"samples_{shape_str}.npz")
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    print('load_models')
    set_seed(101)
    logger.log('decode by rounding. ')
    model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    print('rounding')
    if diffusion.training_mode == 'e2e':
        word_lst = word_lst_e2e
    else:
        word_lst = rounding_func(args.experiment, arr, model, tokenizer, emb_scale_factor=args.emb_scale_factor)

    out_path2 = os.path.join(args.out_dir, f"{model_base_name}.debug_{args.t_start}.txt")
    fout = open(out_path2, 'w')
    out_path3 = os.path.join(args.out_dir, f"{model_base_name}.ref_{args.t_start}.txt")
    foutref  = open(out_path3, 'w')
    judge_lst = []
    for (gg, xx) in zip(ground_truth[0], word_lst):
        # print(tokenizer)
        ref_ = " ".join([tokenizer[y] for y in gg.tolist()])
        print(ref_, file=foutref)
        # print(tokenizer.decode(gg.tolist()))
        # print(gg)
        # print('xx' * 30)
        print(xx, file=fout)
        judge = (xx == ref_)
        judge_lst.append(judge)
        print('---' * 30)
        print(ref_)
        print(xx)
        # print('---' * 30)
    print("the average matching rate is ", np.array(judge_lst).mean())
    fout.close()
    foutref.close()
    print(f'written to {out_path2}')
    print(f'written to {out_path3}')

    dist.barrier()
    logger.log("sampling complete")




def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        model_arch='conv-unet',
        t_start=100,
        # rounding_mode='gpt2',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1)
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
