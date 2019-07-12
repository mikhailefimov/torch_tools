import datetime
import os
import time
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader


class LogSection:
    """
    This class represents a section of the log, related to single "process": epoch, or single training session.
    Do not instantiate it directly, use `Logger.start_session` instead.
    """

    def __init__(self, row_header="", metrics_format="", delimiter="\t", header="",
                 log_steps=True, log_time=True, total=None):
        """
        Args:
            row_header: is printed on each row
            metrics_format: formatting string for metrics in ``str.format`` syntax
            delimiter: delimiter, default - TAB
            header: log section header, printed only once before log rows
            log_steps: if True - steps will be counted, and printed on each row
            log_time: if True - elapsed time will be measured, and printed on each row.
            If total is also specified - estimated time to complete will also be printed
            total: total steps - used to compute estimate time to complete
        """
        self.row_header = row_header
        self.header = header
        self.log_steps = log_steps
        self.log_time = log_time
        self.total = total
        self.cur_step = 0
        self.metrics = defaultdict(list)
        self.metrics_format = metrics_format
        self.delimiter = delimiter

        self.start_time = time.monotonic()
        self.format_str = ""
        self.rebuild_format_str()

    def rebuild_format_str(self):
        format_str = []
        if self.row_header is not None:
            format_str.append(self.row_header)
        if self.log_steps:
            if self.total is None:
                format_str.append('[{iter}/?]')
            else:
                total_str = str(self.total)
                format_str.append('[{iter:' + str(len(total_str)) + 'd}/' + total_str + ']')
        if self.log_time:
            if self.total is None:
                format_str.append('({total_time}s)')
            else:
                format_str.append('({total_time}<={eta_time}s)')
        self.format_str = self.delimiter.join(format_str)
        if self.metrics_format:
            self.format_str = self.format_str + self.delimiter + self.metrics_format

    def update(self, **kwargs):
        r"""Update metrics values for current step.

        Example::

            section.update(loss=0.01, accuracy=0.95)

        """
        for k, v in kwargs.items():
            assert isinstance(v, (float, int))
            metric = self.metrics[k]
            del metric[self.cur_step:]
            while len(metric) < self.cur_step:
                metric.append(None)
            metric.append(v)

    def state_dict(self):
        """Returns a dictionary containing a whole state of this object.
        This dictionary may be persisted and used to call method ``LogSection.from_dict``

        Returns:
            internal state of this section in dict (for persistence)
        """
        return {'metrics': {k: list(v) for k, v in self.metrics.items()},
                'delimiter': self.delimiter, 'metrics_format': self.metrics_format,
                'row_header': self.row_header, 'header': self.header,
                'log_steps': self.log_steps, 'log_time': self.log_time,
                'cur_step': self.cur_step, 'total': self.total}

    @classmethod
    def from_dict(cls, state_dict):
        """Recreates LogSection from dictionary previously obtained from ``state_dict()``

        Args:
            state_dict: dictionary, previously obtained from method state_dict

        Returns:
            new LogSection object, created from given dictionary
        """
        result = cls(row_header=state_dict['row_header'], header=state_dict['header'],
                     delimiter=state_dict['delimiter'], metrics_format=state_dict['metrics_format'],
                     log_steps=state_dict['log_steps'], log_time=state_dict['log_time'],
                     total=state_dict['total'])
        result.cur_step = state_dict['cur_step']
        for k, v in state_dict['metrics'].items():
            result.metrics[k] = list(v)
        return result

    def step_end(self):
        """
        Call this method after each step in process.
        Step counter will be incremented.
        Afterwards all metrics updates will be related to the next step.
        """
        if self.log_time:
            total_time = time.monotonic() - self.start_time
            self.update(total_time=total_time)
            if self.total is not None:
                self.update(eta_time=total_time / (self.cur_step + 1) * (self.total - self.cur_step-1))
        self.cur_step += 1
        for name, metric in self.metrics.items():
            del metric[self.cur_step:]
            while len(metric) < self.cur_step:
                metric.append(None)

    def print_row(self, row=-1, file=None):
        """
        Prints single formatted row from the log

        Args:
            row: index of row to print, zero-based or -1 to print last row
            file: log will be printed to this file or to stdout if None

        """
        values = {}
        for name, metric in self.metrics.items():
            if name == 'total_time' or name == 'eta_time':
                values[name] = str(datetime.timedelta(seconds=int(metric[row])))
            else:
                values[name] = metric[row]

        if self.log_steps:
            values['iter'] = (row + 1 if row >= 0 else self.cur_step)
        print(self.format_str.format(**values), file=file)

    def print_header(self, file=None):
        if self.header:
            print("-" * max(30, len(self.header) + 5), file=file)
            print(self.header, file=file)
            print("-" * max(30, len(self.header) + 5), file=file)

    def print_footer(self, file=None):
        if self.log_time and self.cur_step > 0:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                self.header, str(datetime.timedelta(seconds=int(self.metrics['total_time'][-1]))),
                self.metrics['total_time'][-1] / self.cur_step), file=file)

    def print_logs(self, print_freq=1, file=None):
        """
        Prints full log (header, rows and footer)

        Args:
            print_freq: only last row of each print_freq rows will be printed
            file: log will be printed to this file or to stdout if None
        """
        self.print_header(file=file)
        for row in range(self.cur_step):
            if print_freq and ((row + 1) % print_freq == 0 or row + 1 == self.cur_step):
                self.print_row(row, file=file)
        self.print_footer(file=file)

    def plot_logs(self, metrics=None, title=None):
        """
        Plots graphics for all metrics except `total_time` and `eta_time`

        Args:
            metrics: plot only this metrics (iterable or comma-separated string)
            title: graphic title

        Returns:

        """
        plt.figure(figsize=(19, 6))
        if title is not None:
            plt.suptitle(title)
        if metrics:
            metrics = metrics.split(",") if type(metrics) is str else metrics
        else:
            metrics = list(self.metrics.keys())
            if "total_time" in metrics:
                metrics.remove("total_time")
            if "eta_time" in metrics:
                metrics.remove("eta_time")
        for metric in metrics:
            plt.plot(self.metrics[metric], label=metric)
        plt.legend()


class Logger(object):
    """
    This class may be used to store and optionally print logs for complex multistage processes.
    Each stage is represented by separate LogSection.

    ----

    Example::

        logger = Logger()
        for i in logger.log_every(range(20), 2, "train", metrics_format="loss:{loss:.3f}"):
            logger.update(loss=i*0.01)

        saved_state = logger.state_dict()
        torch.save(saved_state,"logger.file")

        saved_state=torch.load("logger.file")
        loaded_logger = Logger.from_dict(saved_state)
        loaded_logger.print_logs(2)
        loaded_logger.plot_logs()
    """

    def __init__(self):
        self.sections = []

    def state_dict(self):
        """Returns a dictionary containing a whole state of this object.
        This dictionary may be persisted and used to call method ``Logger.from_dict``

        Returns:
            internal state of all section in dict (for persistence)
        """
        return {'sections': [section.state_dict() for section in self.sections]}

    @classmethod
    def from_dict(cls, state_dict):
        """Recreates Logger from dictionary previously obtained from ``state_dict()``

        Args:
            state_dict: dictionary, previously obtained from method state_dict

        Returns:
            new Logger object, created from given dictionary
        """
        result = cls()
        for section in state_dict['sections']:
            log_section = LogSection.from_dict(section)
            result.sections.append(log_section)
        return result

    def update(self, **kwargs):
        r"""Update metrics values for current step in last created section.

        Example::

            logger.update(loss=0.01, accuracy=0.95)

        """
        assert len(self.sections) > 0, "Call log_every() first!"
        self.sections[-1].update(**kwargs)

    def start_section(self, row_header="", header="",
                      metrics_format="", delimiter="\t",
                      log_steps=True, log_time=True,
                      total=None):
        """
        Starts new custom section of log.
        To log processing of iterable object use ``log_every(iterable,...)`` - it will manage section automatically.

        Example::

            custom_section = logger.start_section("custom", "Custom log block without total")
            custom_section.print_header()
            for i in range(10):
                custom_section.step_end()
                custom_section.print_row()
            custom_section.print_footer()


        Args:
            row_header: is printed on each row
            header: log section header, printed only once before log rows
            metrics_format: formatting string for metrics in str.format syntax
            delimiter: delimiter, default - TAB
            log_steps: if True - steps will be counted, and printed on each row
            log_time: if True - elapsed time will be measured, and printed on each row.
                    If total is also specified - estimated time to complete will also be printed
            total: total steps - used to compute estimate time to complete

        Returns:
            new LogSection object

        """
        section = LogSection(row_header=row_header, header=header, metrics_format=metrics_format, delimiter=delimiter,
                             log_steps=log_steps, log_time=log_time, total=total)
        self.sections.append(section)
        return section

    def log_every(self, iterable, print_freq=1, row_header="", header="", metrics_format="", delimiter="\t",
                  log_steps=True, log_time=True, file=None):
        """
        Manages logging of processing ``iterable``.

        Args:
            iterable: iterable or generator to be processed
            print_freq: log row will be printed once per print_freq processed objects
            row_header: is printed on each row (short description of process)
            header: log section header, printed only once before log rows
            metrics_format: formatting string for metrics in str.format syntax
            delimiter: delimiter, default - TAB
            log_steps: if True - steps will be counted, and printed on each row
            log_time: if True - elapsed time and estimated time to complete will be printed on each row.
            file: log will be printed to this file or to stdout if None

        Yields:
            objects obtained from iterable

        """
        section = self.start_section(row_header=row_header, header=header, metrics_format=metrics_format,
                                     delimiter=delimiter, log_steps=log_steps, log_time=log_time, total=len(iterable))
        section.print_header(file=file)
        step = 0
        for obj in iterable:
            yield obj
            section.step_end()
            step += 1
            if print_freq and (step % print_freq == 0 or step == len(iterable)):
                self.sections[-1].print_row(file=file)
        section.print_footer(file=file)

    def print_logs(self, print_freq=1, file=None):
        """
        Prints full log (header, rows and footer) of all sections

        Args:
            print_freq: only last row of each print_freq rows will be printed
            file: log will be printed to this file or to stdout if None
        """
        for section in self.sections:
            section.print_logs(print_freq=print_freq, file=file)

    def plot_logs(self, metrics=None, title=None):
        """
        Plots graphics for all sections

        Args:
            metrics: plot only this metrics (iterable or comma-separated string)
            title: graphics title

        Returns:

        """
        for section in self.sections:
            section.plot_logs(metrics=metrics, title=title)


class ModelBase():
    """
    Base class for wrapping trainable models.
    This class is responsible for:

    - saving and loading model state
    - creating suitable instance of Dataloader
    - feeding model with minibatch data
    - calculating loss and invoking optimizer
    - calculating metrics
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ground_truth = None
        self.predictions = None

    def state_dict(self):
        """This method should return a dictionary containing a whole state of the model"""
        return {'model': self.model.state_dict()}

    def load_state_dict(self, state_dict):
        """This method should restore state of the model from dictionary, previously obtained from state_dict()"""
        self.model.load_state_dict(state_dict['model'])

    def before_epoch(self):
        """This method is called before each training epoch. It must prepare model to training"""
        self.model.train()

    def train_batch(self, X, y, optimizer):
        """
        This is main training method.
        It must run forward and backward passes of training for single minibatch,
        and return loss values and (optionally) other metrics.

        Example::

            prediction = self.model(X.to(self.device))
            loss_value = loss_function(prediction, y.to(self.device))
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            return {"loss":loss_value.item()}

        Args:
            X: first item of tuple, produced by dataloader (input data)
            y: second item of tuple, produced by dataloader (ground truth)
            optimizer: instance of optimizer to be used for backprop training

        Returns:
            dictionary of average loss values per minibatch (and optionally other metrics like train accuracy)
        """
        return {"loss": 0}

    def after_epoch(self):
        """This method is called after each training epoch."""
        pass

    def eval_begin(self, total_size):
        """
        This method is called before evaluation and prediction rounds.
        It must prepare model to evaluation and prepare data structures for accumulating predictions and gt.
        """
        self.model.eval()
        self.ground_truth = []
        self.predictions = []

    def eval_batch(self, X, y):
        """
        This is main evaluation method.
        It must run forward pass for single minibatch, detach and transfer results to cpu
        and convert them to format expected by method eval_append_results()

        Args:
            X: first item of tuple, produced by dataloader (input data)
            y: second item of tuple, produced by dataloader (ground truth for evaluation or None for prediction)

        Returns:
            predicted values for minibatch in format, accepted by method eval_append_results()
        """
        return []

    def eval_append_results(self, predictions, ground_truth):
        """
        This method must append results of single minibatch processing to temporary internal buffer.

        Args:
            predictions: output from eval_batch
            ground_truth: second item of tuple, produced by dataloader (gt for evaluation or None for prediction)

        """
        self.ground_truth.extend(ground_truth)
        self.predictions.extend(predictions)

    def eval_complete(self):
        """
        This method is called after last minibatch in evaluation/prediction round
        It must return collected predictions and ground truth.
        """
        return self.predictions, self.ground_truth

    def evaluate(self, prediction, ground_truth):
        """
        This method must calculate metrics for given prediction and ground truth
        Args:
            prediction: prediction returned by method eval_complete()
            ground_truth:  ground_truth returned by method eval_complete()

        Returns:
            dictionary of calculated metrics values
        """
        return {}

    def dataloader_factory(self, dataset, train, batch_size):
        """
        This method must create DataLoader for given dataset.
        Created DataLoader must produce minibatches in format, suitable for methods train_batch and eval_batch
        Args:
            dataset: dataset
            train: True for training, False for evaluation and prediction.
            batch_size: size of minibatch

        Returns:
            torch.utils.data.DataLoader

        """
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)

    def epoch_metrics_format(self):
        """This method must return format string for printing end of epoch metrics in str.format syntax"""
        return ""

    def batch_metrics_format(self):
        """This method must return format string for printing batch losses and metrics in str.format syntax"""
        return ""

    def target_metric(self):
        """
        This method shold return name of primary metric which should be maximized (used to select best model).
        If this metric should be minimized - prepend it with "-" sign.
        Examples: "acc", "-rmse"
        """
        return None


class Trainer:
    """
    Universal model trainer:

    - train/evaluate/predict loops
    - logging
    - checkpoints persistence
    """

    def __init__(self, model, optimizer_builder, train_dataset,
                 val_dataset="auto", val_split=0.2, batch_size=64, val_batch_size=None,
                 dataloader_factory=None, outdir='.', start_from_checkpoint=None,
                 val_period=1, min_val_score=0, n_best_models=0, verbose=1):
        """

        Args:
            model (ModelBase):
                trainable model
            optimizer_builder((ModelBase) -> torch.optim.Optimizer):
                function, creates optimizer for model
            train_dataset (torch.utils.data.Dataset):
                dataset for training
            val_dataset (torch.utils.data.Dataset or str or None):
                dataset for validation, if "auto" - split part of train_dataset, if None - disable validation
            val_split (double):
                only if val_dataset="auto" - part of train_dataset to be used for validation (part or number of items)
            batch_size (int):
                size of minibatch for training
            val_batch_size (int):
                size of minibatch for evaluation and precition (default =batch_size)
            dataloader_factory ((dataset, train, batch_size)->DataLoader):
                custom replacement for model.dataloader_factory
            outdir (str or None):
                directory for checkpoints, if None - checkpoints are disabled
            start_from_checkpoint (str):
                filename of checkpoint to start from
            val_period (int):
                model will be evaluated once after each val_period epoch (and in the end of training)
            min_val_score:
                minimum acceptable score to start saving best model checkpoints (see also model.target_metric)
            n_best_models:
                maximum number of best models checkpoints to keep in outdir
            verbose:
                print training log row each `verbose` minibatches. If None - report once per epoch
        """
        self.model = model
        if start_from_checkpoint:
            if not os.path.isfile(start_from_checkpoint):
                raise ValueError("start_checkpoint must point to a checkpoint file")
            (model_dict, optimizer_dict, self.total_epochs, self.batch_logger,
             self.epoch_logger, self.last_val_predictions, self.val_gt) = torch.load(start_from_checkpoint)
            self.model.load_state_dict(model_dict)
            self.optimizer = optimizer_builder(self.model)
            self.optimizer.load_state_dict(optimizer_dict)
            if verbose:
                self.batch_logger = self.batch_logger or Logger()
        else:
            self.batch_logger = Logger() if verbose else None
            self.epoch_logger = Logger()
            self.total_epochs = 0
            self.optimizer = optimizer_builder(self.model)

        self.verbose = verbose
        if val_dataset == "auto":
            indices = np.arange(len(train_dataset))
            val_size = int(len(train_dataset) * val_split) if val_split < 1.0 else int(val_split)
            self.train_dataset = Subset(train_dataset, indices[val_size:])
            self.val_dataset = Subset(train_dataset, indices[:val_size])
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset if val_dataset != "none" else None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.dataloader_factory = dataloader_factory or self.model.dataloader_factory
        self.val_period = val_period
        self.min_val_score = min_val_score
        self.n_best_models = n_best_models
        self.best_model_files = []
        self.best_model_score = None
        self.outdir = outdir

        self.target_metric = self.model.target_metric()
        self.target_metric_k = 1.0
        if self.target_metric.startswith("-"):
            self.target_metric = self.target_metric[1:]
            self.target_metric_k = -1.0
        self.val_gt = None
        self.last_val_predictions = None

    def _save_checkpoint(self, name="checkpoint.pickle", best_model=False):
        if self.outdir is None:
            return
        if best_model:
            self.best_model_files.sort(key=itemgetter(1), reverse=True)
            for filename, score in self.best_model_files[self.n_best_models - 1:]:
                if score >= self.best_model_score:
                    return
                try:
                    os.remove(os.path.join(self.outdir, filename))
                except OSError:
                    pass
            self.best_model_files = self.best_model_files[:self.n_best_models - 1]
            self.best_model_files.append((name, self.best_model_score))

        torch.save((self.model.state_dict(), self.optimizer.state_dict(),
                    self.total_epochs, self.batch_logger, self.epoch_logger,
                    self.last_val_predictions, self.val_gt),
                   os.path.join(self.outdir, name))

    def _update_params(self, extra_params, num_epochs):

        scheduler = None

        def set_param(parameter, val):
            if isinstance(val, list) or isinstance(val, tuple):
                for param_group, v in zip(self.optimizer.param_groups, list(val)):
                    param_group[parameter] = v
            else:
                for param_group in self.optimizer.param_groups:
                    param_group[parameter] = val

        for param, value in extra_params.items():
            if param == 'start_lr':
                set_param('initial_lr', value)
                set_param('lr', value)
            if param != 'end_lr':
                set_param(param, value)
        if 'end_lr' in extra_params and num_epochs > 1:
            end_lr = extra_params['end_lr']

            def create_lambda(start, end):
                gamma = (end / start) ** (1 / (num_epochs - 1))
                return lambda epoch: gamma ** epoch

            if isinstance(end_lr, list) or isinstance(end_lr, tuple):
                lambdas = [create_lambda(param_group['lr'], elr)
                           for param_group, elr in zip(self.optimizer.param_groups, list(end_lr))]
            else:
                lambdas = [create_lambda(param_group['lr'], end_lr)
                           for param_group in self.optimizer.param_groups]
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambdas)

        return scheduler

    def _epoch_iterator(self, num_epochs, extra_params):
        return self.epoch_logger.log_every(
            range(num_epochs),
            row_header="Train epoch",
            header="Training epochs %i=>%i  %s" % (
                self.total_epochs, self.total_epochs + num_epochs, repr(extra_params)
            ), metrics_format=self.model.epoch_metrics_format())

    def _batch_iterator(self, data_loader):
        if not self.batch_logger:
            return data_loader
        return self.batch_logger.log_every(data_loader, print_freq=self.verbose, row_header="batch",
                                           metrics_format=self.model.batch_metrics_format())

    @staticmethod
    def _process_metrics_and_losses(metrics_and_losses_accum, metrics_and_losses, y):
        if type(y) is torch.Tensor:
            cur_batch_size = y.size()[0]
        else:
            cur_batch_size = len(y)
        for k in metrics_and_losses.keys():
            v = metrics_and_losses[k]
            if type(v) is torch.Tensor:
                v = v.item()
                metrics_and_losses[k] = v
            metrics_and_losses_accum[k] = metrics_and_losses_accum.get(k, 0) + v * cur_batch_size
        return cur_batch_size

    def train(self, num_epochs, **extra_params):
        """
        Run ``num_epochs`` training epochs. May be executed multiple times.
        Examples::

            train(10)
            train(10, lr=3e-4, weight_decay=1e-5)
            train(10, start_lr=1e-4, end_lr=1e-5)

        Args:
            num_epochs: number of epochs to process
            **extra_params:
                extra parameters for optimizer, like `lr`, `momentum` or `weight_decay`.
                Also accepts special parameters `start_lr' and 'end_lr' for exponential learning rate decay

        Returns:

        """
        scheduler = self._update_params(extra_params, num_epochs)
        data_loader = self.dataloader_factory(self.train_dataset, True, self.batch_size)
        for epoch in self._epoch_iterator(num_epochs, extra_params):
            if epoch > 0:
                self._save_checkpoint()
            metrics_and_losses_accum = {}
            self.model.before_epoch()
            if scheduler is not None:
                scheduler.step(epoch)

            total_samples = 0
            for x, y in self._batch_iterator(data_loader):
                metrics_and_losses = self.model.train_batch(x, y, self.optimizer)
                cur_batch_size = self._process_metrics_and_losses(metrics_and_losses_accum, metrics_and_losses, y)
                total_samples += cur_batch_size
                if self.batch_logger:
                    self.batch_logger.update(**metrics_and_losses)

            metrics_and_losses_avg = {k: (v / total_samples) for k, v in metrics_and_losses_accum.items()}
            metrics_and_losses_avg['learning_rate'] = self.optimizer.param_groups[-1]['lr']

            self.total_epochs += 1

            best_model_score = None
            if (self.val_dataset is not None and
                    (self.val_period > 0 and (self.total_epochs % self.val_period == 0) or (epoch == num_epochs - 1))):
                self.last_val_predictions, self.val_gt, metrics = self.evaluate_model()
                metrics_and_losses_avg.update(metrics)
                if self.target_metric is not None:
                    best_model_score = metrics[self.target_metric] * self.target_metric_k

            self.model.after_epoch()
            self.epoch_logger.update(**metrics_and_losses_avg)
            if best_model_score and (
                    self.best_model_score is not None and best_model_score > self.best_model_score
                    or best_model_score > self.min_val_score):
                self.best_model_score = best_model_score
                self._save_checkpoint("best_model%i.pickle" % self.total_epochs, True)
        self._save_checkpoint()

    def predict(self, dataset):
        """
        Evaluates the model and collects produced predictions and ground truth (if contained in provided dataset)

        Args:
            dataset: dataset for evaluation

        Returns:
            (predictions, ground_truth) tuple as returned by `model.eval_complete()` method
        """
        self.model.eval_begin(len(dataset))
        data_loader = self.dataloader_factory(dataset, False, self.val_batch_size)
        with torch.no_grad():
            for X, y in data_loader:
                batch_predictions = self.model.eval_batch(X, y)
                self.model.eval_append_results(batch_predictions, y)
        predictions, ground_truth = self.model.eval_complete()
        return predictions, ground_truth

    def evaluate_model(self, dataset=None):
        """
        Evaluates the model on a validation dataset and calculates metrics

        Args:
            dataset: this dataset will be used as a validation dataset

        Returns:
            (predictions, ground_truth,metrics) tuple as returned by `model.eval_complete()` and `model.evaluate()`
        """
        dataset = dataset or self.val_dataset
        predictions, ground_truth = self.predict(dataset)
        metrics = self.model.evaluate(predictions, ground_truth)
        return predictions, ground_truth, metrics
