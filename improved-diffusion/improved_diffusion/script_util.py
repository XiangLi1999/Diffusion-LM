import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel
from .transformer_model import TransUNetModel
from .transformer_model2 import TransformerNetModel, TransformerNetModel2
NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        model_arch='trans-unet',
        in_channel=8,
        out_channel=8,
        training_mode='emb',
        vocab_size=66,
        config_name='bert-base-uncased',
        experiment_mode='lm',
        logits_mode=1,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    model_arch,
    in_channel,
    out_channel,
    training_mode,
    vocab_size,
    config_name,
    experiment_mode,
    logits_mode,
    **kwargs,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        model_arch=model_arch,
        in_channel=in_channel,
        out_channel=out_channel,
        training_mode=training_mode,
        vocab_size=vocab_size,
        config_name=config_name,
        experiment_mode=experiment_mode,
        logits_mode=logits_mode,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        model_arch=model_arch,
        training_mode=training_mode,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    model_arch,
    in_channel=8,
    out_channel=8,
    training_mode='emb',
    vocab_size=None,
    config_name='',
    experiment_mode='lm',
    logits_mode=1,
):
    print(f'creating model, based on {model_arch}')
    if model_arch == 'conv-unet':
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16: #DEBUG**
            channel_mult = (1, 2, 2, 2)
        else:
            channel_mult = (1, 2, 2, 2)
        # else:
        #     raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return UNetModel(
            in_channels=in_channel, #3, DEBUG**
            model_channels=num_channels,
            out_channels=(out_channel if not learn_sigma else out_channel*2), # DEBUG**  (3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            training_mode=training_mode,
            vocab_size=vocab_size,
            logits_mode=logits_mode,
        )
    if model_arch == '1d-unet':
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16: #DEBUG**
            channel_mult = (1, 2, 2, 2)
        else:
            channel_mult = (1, 2, 2, 2)
        # else:
        #     raise ValueError(f"unsupported image size: {image_size}")

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return UNetModel(
            in_channels=in_channel, #3, DEBUG**
            model_channels=num_channels,
            out_channels=(out_channel if not learn_sigma else out_channel*2), # DEBUG**  (3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dims=1,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            training_mode=training_mode,
            vocab_size=vocab_size,
            experiment_mode=experiment_mode,
        )
    elif model_arch == 'trans-unet':
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16:  # DEBUG**
            channel_mult = (1, 2, 2, 2)
        else:
            channel_mult = (1, 2, 2, 2)

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return TransUNetModel(
            in_channels=in_channel,  # 3, DEBUG**
            model_channels=num_channels,
            out_channels=(out_channel if not learn_sigma else out_channel*2),  # DEBUG**  (3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            training_mode=training_mode,
            vocab_size=vocab_size,
        )
    elif model_arch == 'transformer':
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16:  # DEBUG**
            channel_mult = (1, 2, 2, 2)
        else:
            channel_mult = (1, 2, 2, 2)

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return TransformerNetModel2(
            in_channels=in_channel,  # 3, DEBUG**
            model_channels=num_channels,
            out_channels=(out_channel if not learn_sigma else out_channel*2),  # DEBUG**  (3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            config_name=config_name,
            training_mode=training_mode,
            vocab_size=vocab_size,
            experiment_mode=experiment_mode,
            logits_mode=logits_mode,
        )
    else:
        raise NotImplementedError

def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    model_arch='conv-unet',
    training_mode='emb',
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if training_mode == 'e2e':
        # end to end training
        if use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE
    elif training_mode == 'e2e-simple':
        if use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE

    else:
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    print(loss_type, learn_sigma)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        model_arch=model_arch,
        training_mode=training_mode,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
