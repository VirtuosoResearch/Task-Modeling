import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from utils import prepare_inputs


def virtual_step(task_to_train_batchs, model, model_, lr, optimizer, task_weights, task_to_index, device):
    """
    Compute unrolled network theta' (virtual step)
    """
    with torch.no_grad():
        for weight, weight_ in zip(model.parameters(), model_.parameters()):
            weight_.copy_(weight_)
    # forward & compute loss
    for task_name, batch in task_to_train_batchs.items():
        batch = prepare_inputs(batch, device)
        outputs = model(**batch, task_name = task_name)
        loss = outputs.loss*task_weights[task_to_index[task_name]]

        # compute gradient
        gradients = torch.autograd.grad(loss, model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(model.parameters(), model_.parameters(), gradients):
                if 'momentum' in optimizer.param_groups[0].keys():  # used in SGD with momentum
                    m = optimizer.state[weight].get('momentum_buffer', 0.) * optimizer.param_groups[0]['momentum']
                else:
                    m = 0
                weight_.data = weight_.data - lr * (m + grad + optimizer.param_groups[0]['weight_decay'] * weight.data)

def compute_hessian(model, model_, d_model, task_to_train_batchs, task_weights, task_to_index, device):
    norm = torch.cat([w.view(-1) for w in d_model]).norm()
    eps = 0.01 / norm

    # \theta+ = \theta + eps * d_model
    with torch.no_grad():
        for p, d in zip(model.parameters(), d_model):
            p += eps * d

    # forward & compute loss
    loss = 0
    for task_name, batch in task_to_train_batchs.items():
        batch = prepare_inputs(batch, device)
        with torch.no_grad():
            outputs = model(**batch, task_name = task_name)
        loss += outputs.loss.item()*task_weights[task_to_index[task_name]]
    d_weight_p = torch.autograd.grad(loss, task_weights)
        
    # \theta- = \theta - eps * d_model
    with torch.no_grad():
        for p, d in zip(model.parameters(), d_model):
            p -= 2 * eps * d

    loss = 0
    for task_name, batch in task_to_train_batchs.items():
        batch = prepare_inputs(batch, device)
        with torch.no_grad():
            outputs = model(**batch, task_name = task_name)
        loss += outputs.loss.item()*task_weights[task_to_index[task_name]]
    d_weight_n = torch.autograd.grad(loss, task_weights)

    # recover theta
    with torch.no_grad():
        for p, d in zip(model.parameters(), d_model):
            p += eps * d

    hessian = [(p - n) / (2. * eps) for p, n in zip(d_weight_p, d_weight_n)]
    return hessian

def update_task_weights(train_data_loaders, valid_data_loaders, model, model_, lr, optimizer, task_weights,
                        task_to_index, target_tasks, device):
    # get a set of train datas and val datas
    task_to_train_batchs = {}; task_to_val_batchs = {}
    for task_name, data_loader in train_data_loaders.items():
        for step, batch in enumerate(data_loader):
            batch = prepare_inputs(batch, device)
            task_to_train_batchs[task_name] = batch
            break

    for task_name, data_loader in valid_data_loaders.items():
        for step, batch in enumerate(data_loader):
            batch = prepare_inputs(batch, device)
            task_to_val_batchs[task_name] = batch
            break

    # do virtual step (calc theta`)
    virtual_step(task_to_train_batchs, model, model_, lr, optimizer, task_weights, task_to_index, device)

    # # define weighting for primary tasks (with binary weights)
    # pri_weights = []
    # for t in tasks:
    #     if t in target_tasks:
    #         pri_weights += [1.0]
    #     else:
    #         pri_weights += [0.0]

    # compute validation data loss on primary tasks
    loss = 0
    for task_name, batch in task_to_val_batchs.items():
        if task_name in target_tasks:
            batch = prepare_inputs(batch, device)
            outputs = model_(**batch, task_name = task_name)
            loss += outputs.loss

    # compute hessian via finite difference approximation
    model_weights_ = tuple(model_.parameters())
    d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
    hessian = compute_hessian(model, model_, d_model, task_to_train_batchs, task_weights, task_to_index, device)

    # update final gradient = - alpha * hessian
    with torch.no_grad():
        for mw, h in zip([task_weights], hessian):
            mw.grad = - lr * h