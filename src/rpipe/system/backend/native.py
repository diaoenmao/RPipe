"""Training / evaluation backend (system layer). Uses explicit RuntimeConfig only."""

from __future__ import annotations

import datetime
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn

from rpipe.algorithm import make_logger
from rpipe.config import RuntimeConfig
from rpipe.data import make_data_loader, process_dataset
from rpipe.model import make_optimizer, make_scheduler
from rpipe.plugins.api import get_provider
from rpipe.schema import assert_valid_result_blob
from rpipe.system import check, resume, save, to_device

cudnn.benchmark = True


class NativeTrainer:
    """Owns the train loop for one (seed, control) run (native PyTorch backend)."""

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime
        self.accelerator = None

    def _build_dataset(self):
        r = self.runtime
        provider = get_provider('data', getattr(r, 'data_provider', 'native'))
        return provider.build(r.data_name)

    def _build_model(self):
        r = self.runtime
        provider = get_provider('model', getattr(r, 'model_provider', 'native'))
        return provider.build(r.model.as_build_dict())

    def _build_logger(self, path: str):
        r = self.runtime
        metric_provider = get_provider('algorithm', getattr(r, 'metric_provider', 'native'))
        metric_obj = None
        if hasattr(metric_provider, 'make_metric'):
            metric_obj = metric_provider.make_metric(r.metric)
        return make_logger(path, **r.log, tag=r.tag, metric=r.metric, metric_obj=metric_obj)

    def _backward(self, loss):
        loss.backward()

    def _prepare(self, model, optimizer, data_loader):
        return model, optimizer, data_loader

    def run(self):
        r = self.runtime
        torch.manual_seed(r.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(r.seed)

        out = r.output_root
        r.path = os.path.join(out, 'exp')
        r.tag_path = os.path.join(r.path, r.tag)
        r.checkpoint_path = os.path.join(r.tag_path, 'checkpoint')
        r.best_path = os.path.join(r.tag_path, 'best')
        r.logger_path = os.path.join(r.tag_path, 'logger', 'train')

        dataset = self._build_dataset()
        process_dataset(dataset, r)
        model = self._build_model()
        result = resume(r.checkpoint_path, resume_mode=r.resume_mode)
        if result is None:
            r.step = 0
            model = model.to(r.device)
            opt_kwargs = r.optimizer.as_kwargs()
            optimizer = make_optimizer(model.parameters(), **opt_kwargs)
            scheduler = make_scheduler(optimizer, opt_kwargs)
            logger = self._build_logger(r.logger_path)
        else:
            cfg_blob = result['cfg']
            r.step = cfg_blob['step'] if isinstance(cfg_blob, dict) else int(getattr(cfg_blob, 'step', 0))
            model = model.to(r.device)
            opt_kwargs = r.optimizer.as_kwargs()
            optimizer = make_optimizer(model.parameters(), **opt_kwargs)
            scheduler = make_scheduler(optimizer, opt_kwargs)
            logger = self._build_logger(r.logger_path)
            model.load_state_dict(result['model'])
            optimizer.load_state_dict(result['optimizer'])
            scheduler.load_state_dict(result['scheduler'])
            logger.load_state_dict(result['logger'])
            logger.reset()

        data_loader = make_data_loader(
            dataset, r.optimizer.batch_size, r.num_steps,
            r.step, r.step_period, r.pin_memory, r.num_workers,
            r.collate_mode, r.seed)
        model, optimizer, data_loader = self._prepare(model, optimizer, data_loader)
        data_iterator = enumerate(data_loader['train'])
        while r.step < r.num_steps:
            self._train_period(data_iterator, model, optimizer, scheduler, logger)
            Evaluator.evaluate_loader(data_loader['test'], model, logger, r, split='test')
            if (r.save_checkpoint or r.step >= r.num_steps) and r.step % r.save_period == 0:
                payload = {
                    'cfg': r.to_dict(),
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'logger': logger.state_dict(),
                }
                check(payload, r.checkpoint_path)
                if logger.compare():
                    shutil.copytree(r.checkpoint_path, r.best_path, dirs_exist_ok=True)
            logger.reset()
        return r.tag_path

    def _train_period(self, data_loader, model, optimizer, scheduler, logger):
        r = self.runtime
        model.train(True)
        start_time = time.time()
        profile = r.log.get('profile') and logger.profiler is not None
        if profile:
            logger.profiler.start()
        for i, input in data_loader:
            if profile:
                logger.profiler.step()
            input_size = input['data'].size(0)
            input = to_device(input, r.device)
            output = model(**input)
            loss = 1 / r.step_period * output['loss']
            self._backward(loss)
            if (i + 1) % r.step_period == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            evaluation = logger.evaluate('train', 'batch', input, output)
            logger.append(evaluation, 'train', n=input_size)
            idx = r.step % r.eval_period
            if idx % max(int(r.eval_period * r.log_interval), 1) == 0 and (i + 1) % r.step_period == 0:
                step_time = (time.time() - start_time) / (idx + 1)
                lr = optimizer.param_groups[0]['lr']
                info = {
                    'info': [
                        'Model: {}'.format(r.tag),
                        'Train Epoch: {}({:.0f}%)'.format(
                            (r.step // r.eval_period) + 1, 100. * idx / r.eval_period),
                        'Learning rate: {:.6f}'.format(lr),
                        'Epoch Finished Time: {}'.format(
                            datetime.timedelta(seconds=round((r.eval_period - (idx + 1)) * step_time))),
                        'Experiment Finished Time: {}'.format(
                            datetime.timedelta(seconds=round((r.num_steps - (r.step + 1)) * step_time))),
                    ]
                }
                logger.append(info, 'train')
                print(logger.write('train'))
            if (i + 1) % r.step_period == 0:
                r.step += 1
            if (idx + 1) % r.eval_period == 0 and (i + 1) % r.step_period == 0:
                break
        if profile:
            logger.profiler.stop()


class Evaluator:
    """Final test pass + result blob for aggregation."""

    def __init__(self, runtime: RuntimeConfig):
        self.runtime = runtime

    def _build_dataset(self):
        r = self.runtime
        provider = get_provider('data', getattr(r, 'data_provider', 'native'))
        return provider.build(r.data_name)

    def _build_model(self):
        r = self.runtime
        provider = get_provider('model', getattr(r, 'model_provider', 'native'))
        return provider.build(r.model.as_build_dict())

    def _build_logger(self, path: str):
        r = self.runtime
        metric_provider = get_provider('algorithm', getattr(r, 'metric_provider', 'native'))
        metric_obj = None
        if hasattr(metric_provider, 'make_metric'):
            metric_obj = metric_provider.make_metric(r.metric)
        return make_logger(path, **r.log, tag=r.tag, metric=r.metric, metric_obj=metric_obj)

    def run(self):
        r = self.runtime
        torch.manual_seed(r.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(r.seed)

        out = r.output_root
        r.path = os.path.join(out, 'exp')
        r.tag_path = os.path.join(r.path, r.tag)
        r.checkpoint_path = os.path.join(r.tag_path, 'checkpoint')
        r.best_path = os.path.join(r.tag_path, 'best')
        r.logger_path = os.path.join(r.tag_path, 'logger', 'test')
        r.result_path = os.path.join(out, 'result', r.tag)

        dataset = self._build_dataset()
        process_dataset(dataset, r)
        model = self._build_model()
        result = resume(r.best_path)
        if result is None:
            raise ValueError('No valid model at {}, please train first'.format(r.best_path))
        if isinstance(result.get('cfg'), dict):
            r.step = result['cfg'].get('step', r.step)
        model = model.to(r.device)
        model.load_state_dict(result['model'])
        data_loader = make_data_loader(
            dataset, r.optimizer.batch_size,
            pin_memory=r.pin_memory, num_workers=r.num_workers,
            collate_mode=r.collate_mode, seed=r.seed)
        test_logger = self._build_logger(r.logger_path)
        self.evaluate_loader(data_loader['test'], model, test_logger, r, split='test')
        ckpt = resume(r.checkpoint_path)
        payload = {
            'schema': 'rpipe.result_blob.v1',
            'cfg': r.to_dict(),
            'logger': {
                'train': ckpt['logger'],
                'test': test_logger.state_dict(),
            },
        }
        assert_valid_result_blob(payload)
        save(payload, r.result_path)
        return r.result_path

    @staticmethod
    def evaluate_loader(data_loader, model, logger, runtime: RuntimeConfig, split='test'):
        with torch.no_grad():
            model.train(False)
            num_steps = len(data_loader)
            if runtime.eval_num_steps != -1:
                num_steps = runtime.eval_num_steps
            input_size = 1
            for i, input in enumerate(data_loader):
                input_size = input['data'].size(0)
                input = to_device(input, runtime.device)
                output = model(**input)
                evaluation = logger.evaluate(split, 'batch', input, output)
                logger.append(evaluation, split, input_size)
                logger.add(split, input, output)
                if (i + 1) == num_steps:
                    break
            evaluation = logger.evaluate(split, 'full')
            logger.append(evaluation, split, input_size)
            info = {
                'info': [
                    'Model: {}'.format(runtime.tag),
                    'Test Epoch: {}({:.0f}%)'.format(runtime.step // runtime.eval_period, 100.),
                ]
            }
            logger.append(info, split)
            print(logger.write(split))
            logger.save()
