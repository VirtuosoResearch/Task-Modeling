from trainer.base_trainer import MultitaskTrainer
from utils import prepare_inputs

import numpy as np
import torch
import torch.nn.functional as F
from utils.attitude import get_ortho_grad_basis, get_low_rank, get_grads_project

class AttitudeTrainer(MultitaskTrainer):

    def __init__(self, model, optimizer, lr_scheduler, config, device, task_to_metrics, multitask_train_data_loader,
        train_data_loaders, valid_data_loaders=None, test_data_loaders=None, 
        task_to_num_labels=None, checkpoint_dir=None, criterion=None,
        target_task = None, collect_gradient_step=1, num_pca_basis = 20, eta_tilde = 1, eta_pos = 1, eta_neg = 0,
        use_target_train = False):
        super().__init__(model, optimizer, lr_scheduler, config, device, task_to_metrics, multitask_train_data_loader,
        train_data_loaders, valid_data_loaders, test_data_loaders, 
        task_to_num_labels, checkpoint_dir, criterion)

        self.collect_gradient_step = collect_gradient_step
        self.num_pca_basis = num_pca_basis
        self.target_task = target_task
        self.max_grad_norm = 1.0
        self.eta_tilde = eta_tilde
        self.eta_pos = eta_pos
        self.eta_neg = eta_neg

        self.target_data_loader = self.train_data_loaders[target_task] if use_target_train else self.valid_data_loaders[target_task] 

    def compute_loss(self, data_loader, task_name, step=1):
        loss = []; count=0
        for _, batch in enumerate(data_loader):
            if count > step:
                break
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss.append(F.cross_entropy(outputs.logits, batch['labels'], reduction="none"))
            count += 1
        return torch.cat(loss, axis=0)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for step, batch in enumerate(self.multitask_train_data_loader):
            if step % 50 == 0:
                # find target task base
                target_loss = self.compute_loss(
                        self.target_data_loader, 
                        task_name=self.target_task, 
                        step = self.collect_gradient_step)
                num_samples = target_loss.size(0)
                ortho_basis = get_ortho_grad_basis(self.model, self.target_task, target_loss, self.device,
                                    num_samples=num_samples, num_pca_basis=self.num_pca_basis)
                
                # get primary task projection
                target_comp, target_grads = get_low_rank(self.model, self.target_task, target_loss, ortho_basis)
                del target_loss, target_grads
            task_name = batch['task_name']
            batch = batch['data']
            
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]

            # iterate other task gradients and do projection
            task_results = get_grads_project(self.model, task_name, loss, ortho_basis, target_comp, 
                self.eta_tilde, self.eta_pos, self.eta_neg)
            task_grads = task_results[-1]

            # update model gradients
            name_params = list(self.model.named_parameters())
            for grad, name_param in zip(task_grads, name_params):
                    name, param = name_param
                    param.grad = (grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # loss.backward()
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