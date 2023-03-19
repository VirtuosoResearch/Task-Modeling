import os
import numpy as np
import numpy
import torch
from numpy import inf
from utils import MetricTracker, prepare_inputs
from collections import OrderedDict
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils import get_logger
from data_loader.multitask_dataset import MultitaskDataset
from metrics.classification import SentenceClassification, margin

class MultitaskTrainer:
    """
    Base class for multitask trainers
    """
    def __init__(self, model, optimizer, lr_scheduler,
                 config, device, task_to_metrics,
                 multitask_train_data_loader,
                 train_data_loaders,
                 valid_data_loaders,
                 test_data_loaders=None, 
                 task_to_num_labels=None,
                 checkpoint_dir=None,
                 criterion=None,
                 eval_margin=False):
        self.logger = get_logger('trainer', config['verbosity'])
        self.device = device

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.cfg_trainer = config
        self.epochs = self.cfg_trainer['num_train_epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.monitor = self.cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max', 'off']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.completed_steps = 0
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir is None:
            self.checkpoint_dir = config["save_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        else:
            for filename in os.listdir(self.checkpoint_dir):
                os.remove(os.path.join(self.checkpoint_dir, filename))

        self.task_to_metrics = task_to_metrics
        self.multitask_train_data_loader = multitask_train_data_loader
        self.train_data_loaders = train_data_loaders
        self.valid_data_loaders = valid_data_loaders
        self.test_data_loaders = test_data_loaders if test_data_loaders is not None else valid_data_loaders
        self.task_to_num_labels = task_to_num_labels
        self.do_validation = self.valid_data_loaders is not None
        self.len_epoch = len(self.train_data_loaders)

        self.is_single_task = len(self.train_data_loaders.keys()) == 1

        self.train_metrics = MetricTracker('loss')
        self.valid_metrics = {task: MetricTracker('loss') for task in self.valid_data_loaders.keys()}
        self.eval_margin = eval_margin
        self._save_checkpoint(epoch=0, name="model_epoch_0")

    def _train_epoch_single_task(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        task_name = list(self.train_data_loaders.keys())[0]
        train_data_loader = self.train_data_loaders[task_name]
        
        for step, batch in enumerate(train_data_loader):
            
            batch = prepare_inputs(batch, self.device)
            outputs = self.model(**batch, task_name = task_name)
            loss = outputs.loss
            loss = loss / self.cfg_trainer["gradient_accumulation_steps"]
            loss.backward()
            if step % self.cfg_trainer["gradient_accumulation_steps"] == 0 or step == len(train_data_loader) - 1:
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

        metric = self.task_to_metrics[task_name]
        if "cdr" in task_name or "youtube" in task_name:
            task_metrics = metric.compute(average = "binary")
        else:
            task_metrics = metric.compute()
        task_metrics = {f'{task_name}_{key}': val for key, val in task_metrics.items()}
        log.update(**task_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

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

        log = self.train_metrics.result()
        for task_name in self.train_data_loaders.keys():
            metric = self.task_to_metrics[task_name]
            if "cdr" in task_name or "youtube" in task_name:
                task_metrics = metric.compute(average = "binary")
            else:
                task_metrics = metric.compute()
            task_metrics = {f'{task_name}_{key}': val for key, val in task_metrics.items()}
            log.update(**task_metrics)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        log = {}
        avg_loss = 0
        for task_name, valid_data_loader in self.valid_data_loaders.items():
            self.valid_metrics[task_name].reset()
            for step, batch in enumerate(valid_data_loader):
                batch = prepare_inputs(batch, self.device)
                outputs = self.model(**batch, task_name = task_name)
                
                predictions = outputs.predictions
                
                self.valid_metrics[task_name].update('loss', outputs.loss.item())
                self.task_to_metrics[task_name].add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )
        
            task_log = self.valid_metrics[task_name].result()
            
            metric = self.task_to_metrics[task_name]
            if "cdr" in task_name or "youtube" in task_name:
                task_metrics = metric.compute(average = "binary")
            else:
                task_metrics = metric.compute()

            task_log.update(task_metrics)
            avg_loss += task_log['loss']

            task_log = {f"{task_name}_{key}": val for key, val in task_log.items()}
            log.update(**task_log)
        log.update(**{'loss': avg_loss/len(self.valid_data_loaders.keys())})
        return log

    def train(self):
        """
        Full training logic
        """
        # do a validation before training
        self.logger.info("At the beginning of training:")
        val_log = self._valid_epoch(epoch=0)
        val_log = {'val_'+k : v for k, v in val_log.items()}
        self.mnt_best = val_log[self.mnt_metric]
        self._save_checkpoint(epoch=0)

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.is_single_task:
                result = self._train_epoch_single_task(epoch)
            else:
                result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            improved = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if improved or self.mnt_mode == 'off':
                self._save_checkpoint(epoch)
        return log

    def test(self, use_valid = False):
        test_data_loaders = self.valid_data_loaders if use_valid else self.test_data_loaders

        if test_data_loaders is None:
            self.logger.info("No test data set.")
            return {}

        # self.model = self.model.from_pretrained(self.checkpoint_dir).to(self.device)
        best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device)["state_dict"])

        log = {}
        avg_loss = 0
        self.model.eval()
        for task_name, test_data_loader in test_data_loaders.items():
            total_loss = 0.0
            margins = []
            for step, batch in enumerate(test_data_loader):
                batch = prepare_inputs(batch, self.device)
                outputs = self.model(**batch, task_name = task_name)
                
                predictions = outputs.predictions
                total_loss += outputs.loss.item() * test_data_loader.batch_size
                
                if self.eval_margin:
                    step_margin = margin(
                            outputs.logits.cpu().detach().numpy(), 
                            batch["labels"].cpu().detach().numpy())
                    margins.append(step_margin)

                self.task_to_metrics[task_name].add_batch(
                    predictions=predictions,
                    references=batch["labels"],
                )

            n_samples = len(test_data_loader.dataset)
            task_log = {'loss': total_loss / n_samples}
            if self.eval_margin:
                task_log.update({'margin': np.mean(margins)})

            metric = self.task_to_metrics[task_name]
            if "cdr" in task_name or "youtube" in task_name:
                task_metrics = metric.compute(average = "binary")
            else:
                task_metrics = metric.compute()
            
            task_log.update(task_metrics)
            avg_loss += task_log['loss']

            task_log = {f"{task_name}_{key}": val for key, val in task_log.items()}
            log.update(**task_log)
        log.update(**{'loss': avg_loss/len(test_data_loaders.keys())})
        self.logger.info(log)
        return log

    def _save_checkpoint(self, epoch, name = "model_best"):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
        }
        # self.logger.info("Best checkpoint in epoch {}".format(epoch))
        
        best_path = os.path.join(self.checkpoint_dir, f'{name}.pth')
        torch.save(state, best_path)
        self.logger.info(f"Saving current model: {name}.pth ...")

    def load_best_model(self, name = "model_best", strict=True):
        # Load the best model then test
        arch = type(self.model).__name__
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        if os.path.exists(best_path):
            state_dict  = torch.load(best_path, map_location=self.device)["state_dict"]
            self.model.load_state_dict(state_dict, strict=strict)
        else: 
            self.logger("No best model checkpoint!")