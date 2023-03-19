import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from utils import prepare_inputs

def get_average_feature_gradients(model, task_name, train_loader, device, step = 1):
    loss = 0
    count = 0
    for i, batch in enumerate(train_loader):
        if i >= step:
            break
        batch = prepare_inputs(batch, device)
        outputs = model(**batch, task_name = task_name)
        loss += outputs.loss
        count += 1
    loss = loss/count
    feature_gradients = grad(loss, model.bert.parameters(), retain_graph=False, create_graph=False,
                             allow_unused=True)
    feature_gradients = torch.cat([gradient.view(-1) for gradient in feature_gradients]) # flatten gradients
    return feature_gradients

def get_task_weights_gradients_multi(model, target_loader, source_loaders, target_task, device, step=1):
    target_gradients = get_average_feature_gradients(model, target_task, target_loader, device, step)

    source_gradients = {}
    for task, task_train_loader in source_loaders.items():
        task_gradients = get_average_feature_gradients(model, task, task_train_loader, device, step)
        source_gradients[task] = task_gradients

    num_tasks = len(source_loaders.keys())
    task_weights_gradients = torch.zeros((num_tasks, ), device=device, dtype=torch.float)
    for i, task in enumerate(source_loaders.keys()):
        task_weights_gradients[i] = -F.cosine_similarity(target_gradients, source_gradients[task], dim=0)
    return task_weights_gradients