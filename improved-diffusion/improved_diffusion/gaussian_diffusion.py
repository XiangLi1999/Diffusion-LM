"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood, discretized_text_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        model_arch=None,
        training_mode='emb',
        # model_arch='conv-unet',
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.model_arch=model_arch

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.training_mode = training_mode
        print('training mode is ', training_mode)
        self.mapping_func = None 
        #
        # if training_mode == 'e2e':
        #     self.training_losses = self.training_losses_e2e
        # else:
        #     self.training_losses = self.training_losses_emb

    def training_losses(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.training_losses_e2e(model, *args, **kwargs)
        elif self.training_mode == 'e2e-simple':
            return self.training_losses_e2e_simple(model, *args, **kwargs)
        else:
            return self.training_losses_emb(model, *args, **kwargs)

    def calc_bpd_loop(self, model, *args, **kwargs):
        if self.training_mode == 'e2e':
            return self.calc_bpd_loop_e2e(model, *args, **kwargs)
        else:
            return self.calc_bpd_loop_emb(model, *args, **kwargs)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance2(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if self.model_arch == 'conv-unet':
            B, C = x.shape[:2]
        else:
            B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)

        # DEBUG:
        if 'debug_x_t' in model_kwargs:
            flag=True
            debug_x_t = model_kwargs.pop('debug_x_t')
            debug_t_batch = model_kwargs.pop('debug_t_batch')
            debug_direct_pred_eps = model_kwargs.pop('debug_direct_pred_eps')
            debug_x_start_cycle_pred = model_kwargs.pop('debug_x_start_cycle_pred')
        else:
            flag=False
        print(model_kwargs)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # DEBUG path:
        def is_very_close(a, b):
            return (((a - b) ** 2).mean())
        direct_pred_eps = model(x, self._scale_timesteps(t), **model_kwargs)
        print(is_very_close(direct_pred_eps, model_output), 'debug 01')
        if flag:
            print(model_kwargs)
            print(is_very_close(debug_direct_pred_eps, model_output), 'debug 001')
            print(is_very_close(debug_x_t, x), 'debug 005')
            print(is_very_close(debug_t_batch.float(), t.float()), 'debug 006')
        x_start_cycle_pred = self._predict_xstart_from_eps(x_t=x, t=t, eps=direct_pred_eps)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if self.model_arch == 'conv-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # print('conv-unet')
            else:
                assert model_output.shape == (B, x.size(1), C * 2)
                model_output, model_var_values = th.split(model_output, C, dim=-1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                print('process_xstart 1')
                x = denoised_fn(x)
            if clip_denoised:
                print('process_xstart 2')
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                print('should go here')
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                print(is_very_close(x_start_cycle_pred, pred_xstart), 'debug 02')
                if flag:
                    print(is_very_close(debug_x_start_cycle_pred, model_output), 'debug 002')
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        print(is_very_close(x_start_cycle_pred, pred_xstart), 'debug 03')
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
            B, C = x.shape[:2]
        else:
            B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        # print(x.shape)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if self.model_arch == 'conv-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # print('conv-unet')
            elif self.model_arch == '1d-unet':
                assert model_output.shape == (B, C * 2, *x.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
            else:
                assert model_output.shape == (B, x.size(1), C * 2)
                model_output, model_var_values = th.split(model_output, C, dim=-1)

            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],
                'greedy_mean':out["mean"], 'out':out}

    def p_debug_loop(self,
                    model,
                    shape,
                    noise=None,
                    clip_denoised=True,
                    denoised_fn=None,
                    model_kwargs=None,
                    device=None,
                    progress=False,):
        final = None
        for sample in self.p_debug_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_debug_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            custom_t_start=100, 
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(custom_t_start))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]


    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                img = out["sample"]

    def p_sample_loop_langevin_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        langevin_func=None,
        top_p=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                if langevin_func is not None:
                    out['t'] = t
                    out['img'] = img 
                    out = langevin_func(out)
                yield out
                img = out["sample"]


    def p_sample_loop_progressive_infill(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
            # img = img[partial_mask] + partial_enc_with_noise[~partial_mask]
        else:
            t_batch = th.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = th.randn(*shape, device=device)
            # print(img.shape, partial_enc_with_noise.shape, partial_mask.shape)
            # img = img[partial_mask] + partial_enc_with_noise[~partial_mask]
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    # img[~partial_mask] = partial_enc_with_noise[~partial_mask]
                    out["sample"] = img
                yield out


    def p_sample_loop_progressive_merge(
        self,
        model,
        shape,
        partial_enc,
        partial_mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        greedy=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
            # img = img[partial_mask] + partial_enc_with_noise[~partial_mask]
        else:
            t_batch = th.tensor([self.num_timesteps - 1] * shape[0], device=device)
            partial_enc_with_noise = self.q_sample(partial_enc, t_batch)
            img = th.randn(*shape, device=device)
            # print(img.shape, partial_enc_with_noise.shape, partial_mask.shape)
            # img = img[partial_mask] + partial_enc_with_noise[~partial_mask]
            img[~partial_mask] = partial_enc_with_noise[~partial_mask]
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if i > 0:
                    partial_enc_with_noise = self.q_sample(partial_enc, t-1)
                else:
                    partial_enc_with_noise = partial_enc
                if greedy:
                    img = out["greedy_mean"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    out["sample"] = img
                else:
                    img = out["sample"]
                    img[~partial_mask] = partial_enc[~partial_mask]
                    # img[~partial_mask] = partial_enc_with_noise[~partial_mask]
                    out["sample"] = img
                yield out

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        top_p=-1.0,
        langevin_fn=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            langevin_fn=langevin_fn,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    langevin_fn=langevin_fn,
                )
                yield out
                img = out["sample"]



    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None,
            noise=None, denoised_fn=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        #lambda *args, r=frozen_out: r,
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        if model_kwargs is not None and 'input_ids' in model_kwargs:
            input_ids = model_kwargs.pop('input_ids')
            mapping_func = model_kwargs.pop('mapping_func', self.mapping_func)
        else:
            input_ids = None
            # noise=None
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        if input_ids is not None:
            # print('input_ids is not None')
            # from torch.distributions import Normal
            # normal_dist = Normal(out["mean"], (0.5 * out["log_variance"]).exp())
            # decoder_nll = -normal_dist.log_prob(x_start)
            assert mapping_func is not None 
            if mapping_func is not None and th.any(t == 0):

                decoder_nll = mapping_func(out["mean"], input_ids) / out["mean"].size(-1)
            else:
                decoder_nll = th.zeros_like(x_start)
            model_kwargs['input_ids'] = input_ids
            model_kwargs['mapping_func'] = mapping_func


            # target = {
            #     ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
            #         x_start=x_start, x_t=x_t, t=t
            #     )[0],
            #     ModelMeanType.START_X: x_start,
            #     ModelMeanType.EPSILON: noise,
            # }[self.model_mean_type]
            # # print(out['mean'].shape, x_start.shape, self.model_mean_type, noise)
            # assert out["mean"].shape == target.shape == x_start.shape
            # decoder_nll = (target - out["mean"]) ** 2
        else:
            decoder_nll = -discretized_gaussian_log_likelihood(
                x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
            )
            assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _vb_terms_bpd_e2e(
            self, model, x_start, x_t, t, input_ids, get_logits, x_start_mean, x_start_log_var, clip_denoised=True,
            model_kwargs=None, noise=None,denoised_fn=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # lambda *args, r=frozen_out: r,
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        assert input_ids is not None
        mapping_func = model_kwargs.pop('mapping_func', self.mapping_func)
        # assert 'input_ids' in model_kwargs
        # input_ids = model_kwargs.pop('input_ids')

        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
        )
        # print(true_log_variance_clipped[0], out["log_variance"][0], 'line1259')
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids) #t=-1

        decoder_nll = decoder_nll / out["mean"].size(-1)
        decoder_nll = decoder_nll / np.log(2.0)


        mask_1 = (t == 0)
        if mask_1.any():
            # print(x_start_log_var, out["log_variance"][0], 't=0')
            # kl_T = normal_kl(
            #     x_start_mean, x_start_log_var, out["mean"], x_start_log_var #out["log_variance"]
            # )
            kl_T = normal_kl(
                x_start_mean, x_start_log_var, out["mean"], out["log_variance"]
            )
            kl_T = mean_flat(kl_T) / np.log(2.0)
            # print('1111',kl_T.shape, mask_1, kl.shape)
            kl = th.where(mask_1, kl_T, kl)

        out_mean, out_variance, \
        out_log_variance_clipped = self.q_mean_variance(x_start,
                                                        th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        kl_T = normal_kl(
            out_mean, out_log_variance_clipped, 0, 0
        )
        kl_T = mean_flat(kl_T) / np.log(2.0)

        # print(decoder_nll, )
        # print()
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        # output = th.where((t == 0), decoder_nll, kl)
        output = kl + decoder_nll + kl_T 
        return {"output": output, "pred_xstart": out["pred_xstart"], 'kl': kl, 'decoder_nll':decoder_nll, 'kl_T':kl_T}

    def training_losses_emb(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                noise=noise,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                # print(terms["loss"])
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if model_kwargs is not None and 'input_ids' in model_kwargs:
                model_kwargs.pop('input_ids')
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                if self.model_arch == 'conv-unet':
                    B, C = x_t.shape[:2]
                elif self.model_arch == '1d-unet':
                    B, C = x_t.size(0), x_t.size(1)
                else:
                    B, C = x_t.size(0), x_t.size(-1)
                # B, C = x_t.shape[:2]

                if self.model_arch == 'conv-unet':
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                    # print('conv-unet')
                elif self.model_arch == '1d-unet':
                    # print(model_output.shape, (B, C * 2, *x_t.shape[2:]))
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                else:
                    # print(model_output.shape, (B, x.size(1), C * 2), x.shape, 'gaussian diffusion.')
                    assert model_output.shape == (B, x_t.size(1), C * 2)
                    model_output, model_var_values = th.split(model_output, C, dim=-1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=-1)

                # assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                # model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.


                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    noise=noise,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise = th.randn_like(x_start_mean)
        # print(std.shape, noise.shape, x_start_mean.shape)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )

    def token_discrete_loss(self, x_t, get_logits, input_ids):
        if self.model_arch == 'conv-unet' or  self.model_arch == '1d-unet':
            reshaped_x_t = x_t.view(x_t.size(0), x_t.size(1), -1).permute(0, 2, 1)
        else:
            # print(x_t.shape)
            reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        # print(decoder_nll.shape)
        decoder_nll = decoder_nll.mean(dim=-1)
        return decoder_nll

    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart =  self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            pred_prev = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            raise NotImplementedError(self.model_mean_type)
        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}


    def training_losses_e2e(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        assert 'input_ids' in model_kwargs
        input_ids = model_kwargs.pop('input_ids').to(t.device)
        x_start_mean = model.model.module.get_embeds(input_ids)
        if self.model_arch == 'conv-unet':
            seqlen = int(np.sqrt(input_ids.size(1)))
            x_start_mean = x_start_mean.view(x_start_mean.size(0), seqlen, seqlen, x_start_mean.size(-1)).permute(0, 3,
                                                                                                                  1, 2)
        elif self.model_arch == '1d-unet':
            x_start_mean = x_start_mean.permute(0, 2, 1)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        # print(std.shape, )
        x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        # print(x_start_mean.shape, x_start.shape)
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # reparametrization trick.
        get_logits = model.model.module.get_logits

        terms = {}

        if self.loss_type == LossType.E2E_KL:
            terms["loss"] = self._vb_terms_bpd_e2e(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                input_ids=input_ids,
                get_logits=get_logits,
                x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                noise=noise,
            )["output"]
            # if self.loss_type == LossType.RESCALED_KL:
            terms["loss"] *= self.num_timesteps


        elif self.loss_type == LossType.E2E_MSE or self.loss_type == LossType.E2E_RESCALED_MSE:
            # print(x_t.shape)
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
                    B, C = x_t.shape[:2]
                else:
                    B, C = x_t.size(0), x_t.size(-1)

                if self.model_arch == 'conv-unet':
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                    # print('conv-unet')
                else:
                    # print(model_output.shape, (B, x.size(1), C * 2), x.shape, 'gaussian diffusion.')
                    assert model_output.shape == (B, x_t.size(1), C * 2)
                    model_output, model_var_values = th.split(model_output, C, dim=-1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=-1)


                terms["vb"] = self._vb_terms_bpd_e2e(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=False,
                    noise=noise,
                )["output"]
                if self.loss_type == LossType.E2E_RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            # print( terms["mse"])
            model_out_x_start = self.x0_helper(model_output, x_t, t)['pred_xstart']
            t0_mask = (t == 0)
            t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
            # print(terms["mse"].shape, )
            terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

            # tT_mask = (t == self.num_timesteps - 1)
            out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
            tT_loss =  mean_flat(out_mean ** 2)

            decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids)

            # assert (model.lm_head.weight == model.word_embedding.weight).all()

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                # KEY
                terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)
                # terms["loss"] = terms["mse"] + (1.0/self.num_timesteps) * decoder_nll + \
                #                 (1.0/self.num_timesteps) * tT_loss
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def training_losses_e2e_simple(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        assert 'input_ids' in model_kwargs
        x_start = None
        input_ids = model_kwargs.pop('input_ids').to(t.device)
        x_start_mean = model.model.module.get_embeds(input_ids)
        if self.model_arch == 'conv-unet':
            seqlen = int(np.sqrt(input_ids.size(1)))
            x_start_mean = x_start_mean.view(x_start_mean.size(0), seqlen, seqlen, x_start_mean.size(-1)).permute(0, 3,
                                                                                                                  1, 2)
        elif self.model_arch == '1d-unet':
            x_start_mean = x_start_mean.permute(0, 2, 1)
        x_start = x_start_mean
        # print(x_start_mean.shape, x_start.shape)
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise) # reparametrization trick.
        get_logits = model.model.module.get_logits

        terms = {}

        if self.loss_type == LossType.E2E_Simple_KL:
            raise NotImplementedError
            terms["loss"] = self._vb_terms_bpd_e2e(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                input_ids=input_ids,
                get_logits=get_logits,
                x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                noise=noise,
            )["output"]
            # if self.loss_type == LossType.RESCALED_KL:
            terms["loss"] *= self.num_timesteps


        elif self.loss_type == LossType.E2E_Simple_MSE:
            # print('simple mse training ')
            # print(x_t.shape)
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                if self.model_arch == 'conv-unet' or self.model_arch == '1d-unet':
                    B, C = x_t.shape[:2]
                else:
                    B, C = x_t.size(0), x_t.size(-1)

                if self.model_arch == 'conv-unet':
                    assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                    # print('conv-unet')
                else:
                    # print(model_output.shape, (B, x.size(1), C * 2), x.shape, 'gaussian diffusion.')
                    assert model_output.shape == (B, x_t.size(1), C * 2)
                    model_output, model_var_values = th.split(model_output, C, dim=-1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=-1)


                terms["vb"] = self._vb_terms_bpd_e2e(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=False,
                    noise=noise,
                )["output"]
                if self.loss_type == LossType.E2E_RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            # terms["mse"] = mean_flat((target - model_output) ** 2)

            ce = th.nn.CrossEntropyLoss(reduction='none')
            model_logits = get_logits(model_output)
            # print(model_logits.shape)
            ce_loss = ce(model_logits.view(-1, model_logits.size(-1)), input_ids.view(-1))
            ce_loss = ce_loss.view(input_ids.shape)
            # print(ce_loss.shape)
            terms["ce"] = mean_flat(ce_loss)
            # print(terms["ce"].shape)

            # out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device))
            # tT_loss =  mean_flat(out_mean ** 2)

            # decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids) / model_output.size(-1) #DEBUG, only true for transformer model.

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                # KEY
                # terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)
                terms["loss"] = terms["ce"] # + tT_loss

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop_e2e(self, model, x_start, clip_denoised=True, model_kwargs=None, denoised_fn=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        input_ids = model_kwargs.pop('input_ids').to(device)
        x_start_mean = model.get_embeds(input_ids)
        if self.model_arch == 'conv-unet':
            seqlen = int(np.sqrt(input_ids.size(1)))
            x_start_mean = x_start_mean.view(x_start_mean.size(0), seqlen, seqlen, x_start_mean.size(-1)).permute(0, 3,
                                                                                                                  1, 2)
        elif self.model_arch == '1d-unet':
            x_start_mean = x_start_mean.permute(0, 2, 1)
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        x_start_log_var = 2 * th.log(std)
        # print(std)
        x_start = self.get_x_start(x_start_mean, std)
        get_logits = model.get_logits

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd_e2e(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean, x_start_log_var=x_start_log_var,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            if t == self.num_timesteps -1:
                assert len(vb) == 0
                vb.append(out["kl_T"])
            vb.append(out["kl"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))
        vb.append(out["decoder_nll"])

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        # prior_bpd = self._prior_bpd(x_start)
        prior_bpd = out["kl_T"]
        total_bpd = vb.sum(dim=1)
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


    def calc_bpd_loop_emb(self, model, x_start, clip_denoised=True, model_kwargs=None,
                          denoised_fn=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            # print(t)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])

            #
            # ## DEBUG
            # def is_very_close(a, b):
            #     return (((a - b) ** 2).mean())
            # x_start_cycle = self._predict_xstart_from_eps(x_t=x_t, t=t_batch, eps=noise)
            # gold_eps_cycle = self._predict_eps_from_xstart(x_t, t_batch, x_start_cycle)
            # print(((gold_eps_cycle-noise)**2).mean())

            # print(is_very_close(out2['pred_xstart'],out["pred_xstart"]), 'first isclose --> check p_mean')
            # model.eval()
            # with th.no_grad():
            #     direct_pred_eps = model(x_t, self._scale_timesteps(t_batch), **model_kwargs)
                # print(((direct_pred_eps - noise) ** 2).mean(), 'ans1', self.rescale_timesteps)

                # x_start_cycle_pred = self._predict_xstart_from_eps(x_t=x_t, t=t_batch, eps=direct_pred_eps)
                # model_kwargs['debug_x_t'] = x_t
                # model_kwargs['debug_t_batch'] = t_batch
                # model_kwargs['debug_direct_pred_eps'] = direct_pred_eps
                # model_kwargs['debug_x_start_cycle_pred'] = x_start_cycle_pred

                # out2 = self.p_mean_variance(
                #     model, x_t, t_batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
                # )
                # # print(((out["pred_xstart"] - x_start_cycle_pred) ** 2).mean(), 'if not align issue with vb_terms')
                # print(is_very_close(out2['pred_xstart'], x_start_cycle_pred), '2nd isclose --> check our flattened')
                # gold_eps_cycle_pred = self._predict_eps_from_xstart(x_t, t_batch, x_start_cycle_pred)

                # print(((eps - noise) ** 2).mean(), 'ans2', self._scale_timesteps)
                # print()
            # print(((gold_eps_cycle_pred - direct_pred_eps) ** 2).mean(), 'should be same, exactly same computation..')
            ## DEBUG
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
