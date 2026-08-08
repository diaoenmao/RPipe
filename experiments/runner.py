from __future__ import annotations

from rpipe.config import (
    apply_control_name,
    build_runtime_cfg,
    experiment_from_mapping,
    load_default_dict,
)
from rpipe.system.backend import Evaluator, Trainer
import rpipe.provider  # noqa: F401 — register layer providers
from experiments.artifacts import write_run_manifest
from experiments.prepare import prepare_datasets
from experiments.process import process_suite_results
from experiments.suite import expand_controls, load_suite

STAGES = ('prepare', 'train', 'test', 'process', 'artifacts')


class ResearchPipeline:
    """End-to-end research loop over a named experiment suite."""

    def __init__(self, suite_name: str, device: str | None = None, stages=None, force_prepare=False,
                 output_root: str = 'output'):
        self.suite = load_suite(suite_name)
        self.device = device or 'cpu'
        self.stages = self._parse_stages(stages)
        self.force_prepare = force_prepare
        self.output_root = output_root
        self.notes: list[str] = []
        self.result_paths: list[str] = []
        self.manifest_path: str | None = None
        self.base_exp = experiment_from_mapping(load_default_dict(), hyper=self.suite.get('hyper'))
        self.base_exp.device = self.device
        self.base_exp.output_root = output_root
        if self.device == 'cpu':
            self.base_exp.pin_memory = False
        for key in (
            'data_provider', 'model_provider',
            'train_algorithm', 'metric_algorithm', 'generate_algorithm',
            'system_provider', 'mixed_precision',
        ):
            if key in self.suite:
                setattr(self.base_exp, key, self.suite[key])
            elif key in (self.suite.get('hyper') or {}):
                setattr(self.base_exp, key, self.suite['hyper'][key])

    @staticmethod
    def _parse_stages(stages):
        if stages is None:
            return list(STAGES)
        if isinstance(stages, str):
            stages = [s.strip() for s in stages.split(',') if s.strip()]
        stages = ['artifacts' if s == 'report' else s for s in stages]
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError('Unknown stages: {}. Allowed: {}'.format(unknown, STAGES))
        return list(stages)

    def run(self):
        print('[pipeline] suite={} stages={} device={}'.format(
            self.suite['name'], ','.join(self.stages), self.device))
        if 'prepare' in self.stages:
            prepare_datasets(
                self.suite['data_names'],
                output_root=self.output_root,
                force=self.force_prepare,
                device=self.device,
            )
        if 'train' in self.stages:
            self._stage_train()
        if 'test' in self.stages:
            self._stage_test()
        if 'process' in self.stages:
            self._stage_process()
        if 'artifacts' in self.stages:
            self.manifest_path = write_run_manifest(
                self.suite,
                stages=self.stages,
                device=self.device,
                result_paths=self.result_paths,
                notes=self.notes,
                output_root=self.output_root,
            )
        return {
            'suite': self.suite['name'],
            'stages': self.stages,
            'manifest_path': self.manifest_path,
            'notes': self.notes,
        }

    def _runtime_for(self, control_name: str, seed: str):
        exp = apply_control_name(self.base_exp, control_name)
        exp.device = self.device
        exp.output_root = self.output_root
        exp.hyper_overrides = dict(self.suite.get('hyper') or {})
        for key, value in exp.hyper_overrides.items():
            if hasattr(exp.train, key):
                setattr(exp.train, key, value)
        return build_runtime_cfg(exp, seed=int(seed))

    def _stage_train(self):
        for seed, control_name in expand_controls(self.suite):
            runtime = self._runtime_for(control_name, seed)
            print('[train] {}'.format(runtime.tag))
            Trainer(runtime).run()

    def _stage_test(self):
        for seed, control_name in expand_controls(self.suite):
            runtime = self._runtime_for(control_name, seed)
            print('[test] {}'.format(runtime.tag))
            path = Evaluator(runtime).run()
            self.result_paths.append(path)

    def _stage_process(self):
        seeds = [str(s) for s in range(
            self.suite['init_seed'],
            self.suite['init_seed'] + self.suite['num_experiments'])]
        controls = expand_controls(self.suite)
        process_suite_results(controls, seeds, output_root=self.output_root)
