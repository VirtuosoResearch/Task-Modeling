from trainer.base_trainer import MultitaskTrainer
from utils import prepare_inputs

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad


import os
import json

class TAGTrainer(MultitaskTrainer):

    def __init__(self, model, optimizer, lr_scheduler, config, device, task_to_metrics, 
        multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders=None, 
        task_to_num_labels=None, checkpoint_dir=None, criterion=None,
        target_task = None, use_target_train = False, collect_gradient_step = 10, update_lr = 1e-4, record_step = 10,
        affinity_dir = '', affinity_type='tag'):
        super().__init__(model, optimizer, lr_scheduler, config, device, task_to_metrics, 
        multitask_train_data_loader, train_data_loaders, valid_data_loaders, test_data_loaders, 
        task_to_num_labels, checkpoint_dir, criterion)

        self.target_task = target_task
        self.use_target_train = use_target_train # whether the train dataloader of the target task is in self.train_data_loaders
        self.target_data_loader = self.train_data_loaders[target_task] if use_target_train else self.valid_data_loaders[target_task] 
        self.collect_gradient_step = collect_gradient_step
        self.update_lr = update_lr

        tasks = list(self.train_data_loaders.keys()) if use_target_train else [self.target_task,] + list(self.train_data_loaders.keys())
        self.task_gains = {task: dict([(t, []) for t in tasks]) for task in tasks}
        self.affinity_dir = affinity_dir

        self.global_step = 0
        self.record_step = record_step

        self.affinity_type = affinity_type

    def compute_loss_and_gradients(self, data_loader, task_name, step=1):
        loss = 0; count=0
        for _, batch in enumerate(data_loader):
            if count > step:
                break
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss += outputs.loss
            count += 1
        loss = loss/count
        feature_gradients = grad(loss, self.model.bert.parameters(), retain_graph=False, create_graph=False,
                             allow_unused=True)
        return loss.cpu().item(), feature_gradients

    def update_task_gains(self, step_gains):
        for task, task_step_gain in step_gains.items():
            for other_task in task_step_gain.keys():
                self.task_gains[task][other_task].append(task_step_gain[other_task])
        
    def save_task_gains(self):
        for task, gains in self.task_gains.items():
            for other_task in gains.keys():
                gains[other_task] = np.mean(gains[other_task])
        
        # save the task affinity
        with open(self.affinity_dir, "w") as f:
            task_affinity = json.dumps(self.task_gains)
            f.write(task_affinity)
        self.logger.info(task_affinity)

    def compute_tag_task_gains(self):
        task_gain = {task: dict() for task in self.train_data_loaders.keys()}
        if not self.use_target_train:
            task_gain.update({self.target_task: dict()})

        # 1. collect task losses
        if self.use_target_train:
            task_losses = {}
            task_gradients = {}
        else:
            target_loss, target_gradients = self.compute_loss_and_gradients(self.target_data_loader, self.target_task, self.collect_gradient_step)
            task_losses = {self.target_task: target_loss}
            task_gradients = {self.target_task: target_gradients}

        for task, train_data_loader in self.train_data_loaders.items():
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(train_data_loader, task, self.collect_gradient_step)
            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients

        task_data_loaders = {key: val for key, val in self.train_data_loaders.items()}
        if not self.use_target_train:
            task_data_loaders.update({self.target_task: self.target_data_loader})

        for task in task_data_loaders.keys():
            # 2. take a gradient step on the task loss
            encoder_weights = list(self.model.bert.parameters())
            encoder_gradients = task_gradients[task]
            for i, weight in enumerate(encoder_weights):
                weight.data -= encoder_gradients[i].data * self.update_lr

            # 3. evaluate losses on the target task
            other_tasks =  task_data_loaders.keys() if self.use_target_train else [self.target_task]
            for other_task in other_tasks:
                update_loss, _ = self.compute_loss_and_gradients(task_data_loaders[other_task], other_task, self.collect_gradient_step)
                task_gain[other_task][task] =  1 - update_loss/task_losses[other_task]

            # 4. restore weights
            for i, weight in enumerate(encoder_weights):
                weight.data += encoder_gradients[i].data * self.update_lr

        return task_gain

    def compute_cs_task_gains(self):
        task_gain = {task: dict() for task in self.train_data_loaders.keys()}
        if not self.use_target_train:
            task_gain.update({self.target_task: dict()})

        # 1. collect task losses and gradients
        if self.use_target_train:
            task_losses = {}
            task_gradients = {}
        else:
            target_loss, target_gradients = self.compute_loss_and_gradients(self.target_data_loader, self.target_task, self.collect_gradient_step)
            target_gradients = torch.cat([gradient.view(-1) for gradient in target_gradients])
            
            task_losses = {self.target_task: target_loss}
            task_gradients = {self.target_task: target_gradients}

        for task, train_data_loader in self.train_data_loaders.items():
            tmp_loss, tmp_gradients = self.compute_loss_and_gradients(train_data_loader, task, self.collect_gradient_step)
            tmp_gradients = torch.cat([gradient.view(-1) for gradient in tmp_gradients])
            
            task_losses[task] = tmp_loss
            task_gradients[task] = tmp_gradients

        task_data_loaders = {key: val for key, val in self.train_data_loaders.items()}
        if not self.use_target_train:
            task_data_loaders.update({self.target_task: self.target_data_loader})

        # 2. compute cosine similarity
        other_tasks =  task_data_loaders.keys() if self.use_target_train else [self.target_task]
        for other_task in other_tasks:
            for task in task_data_loaders.keys():
                task_gain[other_task][task] = F.cosine_similarity(
                    task_gradients[other_task], task_gradients[task], dim=0
                ).cpu().item()

        return task_gain

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for step, batch in enumerate(self.multitask_train_data_loader):
            task_name = batch['task_name']
            batch = batch['data']
            
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(self.multitask_train_data_loader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.completed_steps += 1

            # update training metrics
            self.train_metrics.update('loss', loss.item())
            
            predictions = outputs.predictions
            labels = batch["labels"]
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            self.task_to_metrics[task_name].add_batch(
                predictions=predictions,
                references=labels,
            )

            if self.completed_steps > self.cfg_trainer["max_train_steps"]:
                break
            
            # compute iter-task affinity
            if self.global_step % self.record_step == 0:
                if self.affinity_type == 'tag':
                    step_task_gains = self.compute_tag_task_gains()
                elif self.affinity_type == 'cs':
                    step_task_gains = self.compute_cs_task_gains()
                self.update_task_gains(step_task_gains)
            self.global_step += 1

        log = self.train_metrics.result()
        for task_name in self.train_data_loaders.keys():
            metric = self.task_to_metrics[task_name]
            task_metrics = metric.compute()
            task_metrics = {f'{task_name}_{key}': val for key, val in task_metrics.items()}
            log.update(**task_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log
