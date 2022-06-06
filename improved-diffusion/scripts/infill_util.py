import torch as th

def get_score(input_embs, label_ids, model_control, t=None):
    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    # print(label_ids2[:, 65:])
    # print(final.shape, tgt_embs.shape)
    # input_embs = th.cat([final, tgt_embs], dim=1)
    model_out = model_control(input_embs=input_embs,
                              labels=label_ids2, t=t)
    print(model_out.loss, 'final end')
    loss_fn = th.nn.CrossEntropyLoss(reduction='none')
    shifted_logits = model_out.logits[:, :-1].contiguous()
    shifted_labels = label_ids2[:, 1:].contiguous()
    loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)).reshape(
        shifted_labels.shape)
    return loss.sum(dim=-1).tolist()


def langevin_fn3(debug_lst, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    # K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    tgt_embs = model3(label_ids[:, sample.size(1):])

    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    input_embs_param = th.nn.Parameter(sample)
    if False:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            model_out = model_control(input_embs=input_embs,
                                      labels=label_ids2, t=tt)

            coef = 0.01
            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            # print(model_out.loss, f'start_{i}', logp_term.item(), t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
    # model_out = model_control(input_embs=input_embs,
    #                           labels=label_ids2,
    #                           t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn4(debug_lst, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    input_embs_param = th.nn.Parameter(sample)
    if False:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            # print(input_embs_param.shape, label_ids.shape)
            model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)

            coef = 0.0001 # prev default.
            # coef = 0.001
            # coef = 0.0005


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
            print(model_out.loss, f'start_{i}', logp_term.item(),
                  t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
    print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn_length(coeff, diffusion, partial_mask, diff_model, tgt_embs, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    input_embs_param = th.nn.Parameter(sample)
    if False:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            print(t.shape)
            # print(input_embs_param.shape, label_ids.shape)
            out = diffusion.p_mean_variance(
                diff_model,
                input_embs_param,
                t,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs={},
            )

            # model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
            coef = coeff
            # coef = 0.0001 # prev default.
            # coef = 0.001
            # coef = 0.0005


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                infill_loss = (out['pred_xstart'][~partial_mask] - tgt_embs[~partial_mask]) ** 2
                infill_loss = infill_loss.mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                # print(out['pred_xstart'].shape, tgt_embs.shape)
                # print(partial_mask[0])
                infill_loss = ((out['pred_xstart'][~partial_mask] - tgt_embs[~partial_mask]) ** 2).view(tgt_embs.size(0), -1, tgt_embs.size(-1) )
                # print(infill_loss.shape, ((mean - input_embs_param)**2).shape )
                infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            print(infill_loss, f'start_{i}', logp_term.item(),
                  t[0].item(), sigma.mean().item())
            loss = logp_term + infill_loss
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # model_out = model_control(input_embs=input_embs_param, pos_ids=label_ids, t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn_tree(coeff, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample): # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3

    if t[0].item() >0:
        tt =t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    input_embs_param = th.nn.Parameter(sample)

    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            # print(input_embs_param.shape, label_ids.shape)
            model_out = model_control(input_embs=input_embs_param, parse_chart=label_ids, t=tt)

            # coef = 0.0001
            # coef = 0.001
            # coef = 0.01

            # coef = 0.1 # good for partial.
            # coef=0.001 # also good for full (more fluent).
            # coef=0.0001

            # coef=0.0005 # good for full.
            coef = coeff

            # coef = 0.5


            # coef=1.
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
            # print(model_out.loss, f'start_{i}', logp_term.item(),
            #       t[0].item(), sigma.mean().item())
            loss = model_out.loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    # COMMENT OUT 
    # model_out = model_control(input_embs=input_embs_param, parse_chart=label_ids, t=tt)
    # print(model_out.loss, 'end')

    return input_embs_param.data

def langevin_fn1(debug_lst, model_control, model3, label_ids, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 1
    # K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200
    label_ids = label_ids.cuda()
    tgt_embs = model3(label_ids[:, sample.size(1):])

    label_ids2 = label_ids.clone()
    label_ids2[:, :65] = -100
    input_embs_param = th.nn.Parameter(sample)
    if True:
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            model_out = model_control(input_embs=input_embs,
                                      labels=label_ids2, t=tt)

            # coef = 0.0
            # if sigma.mean() == 0:
            #     logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            # else:
            #     logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            print(model_out.loss, f'start_{i}', t[0].item(), sigma.mean().item())
            coef = 3.
            loss = model_out.loss # + logp_term
            loss.backward()
            # print(input_embs_param.grad.shape, )
            input_embs_param.data = input_embs_param.data - coef * sigma.mean().item() * input_embs_param.grad
            # optimizer.step()
            # epsilon = th.randn_like(input_embs_param.data)
            # input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())
            # input_embs_param = th.nn.Parameter((input_embs_param.data +
            #                                    np.sqrt(2*sigma.mean().item()) * epsilon).detach())

    input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
    model_out = model_control(input_embs=input_embs,
                              labels=label_ids2,
                              t=tt)
    print(model_out.loss, 'end')
    # if True:
    #     debug_lst.append(get_score(input_embs, label_ids2, model_control, t=tt))

    return input_embs_param.data


def langevin_fn3_compose(debug_lst, model_control, model3, label_ids_lst, step_size, sample, mean, sigma,
                 alpha, t, prev_sample):  # current best.
    if t[0].item() < 10:
        K = 0
    else:
        K = 3
    # K = 3

    if t[0].item() > 0:
        tt = t[0].item() - 1
    else:
        tt = 200

    tgt_embs_lst = [model3(label_ids[:, sample.size(1):]) for label_ids in label_ids_lst]

    label_ids2_lst = []
    for label_ids in label_ids_lst:
        label_ids2 = label_ids.clone()
        label_ids2[:, :65] = -100
        label_ids2_lst.append(label_ids2)

    input_embs_param = th.nn.Parameter(sample)
    if True:
        part_score = []
        for (tgt_embs,label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
            input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
            score_ = get_score(input_embs, label_ids2, model_control, t=tt)
            part_score.append(score_)
        debug_lst.append(part_score)
    with th.enable_grad():
        for i in range(K):
            optimizer = th.optim.Adagrad([input_embs_param], lr=step_size)
            optimizer.zero_grad()
            cum_loss = 0
            for (tgt_embs, label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
                input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
                model_out = model_control(input_embs=input_embs,
                                          labels=label_ids2, t=tt)
                cum_loss += model_out.loss

            coef = 0.01
            if sigma.mean() == 0:
                logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
            else:
                logp_term = coef * ((mean - input_embs_param) ** 2 / sigma).mean(dim=0).sum()
            print(cum_loss, f'start_{i}', logp_term.item(), t[0].item(), sigma.mean().item())
            loss = cum_loss + logp_term
            loss.backward()
            optimizer.step()
            epsilon = th.randn_like(input_embs_param.data)
            input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0 * sigma.mean().item() * epsilon).detach())

    part_score = []
    for (tgt_embs, label_ids2) in zip(tgt_embs_lst, label_ids2_lst):
        input_embs = th.cat([input_embs_param, tgt_embs], dim=1)
        score_ = get_score(input_embs, label_ids2, model_control, t=tt)
        part_score.append(score_)

    return input_embs_param.data
