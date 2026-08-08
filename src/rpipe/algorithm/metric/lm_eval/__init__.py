"""algorithm/metric/lm_eval — PyPI ``lm_eval`` (benchmark metrics)."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register_algorithm


@register_algorithm('metric', 'lm_eval')
class LMEvalMetricAlgorithm:
    name = 'lm_eval'
    package = 'lm_eval'
    algorithm_type = 'metric'
    mode = 'benchmark'

    def available(self) -> bool:
        try:
            import lm_eval  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return ['gsm8k', 'mmlu', 'hellaswag', 'arc_challenge', 'humaneval']

    def list_tasks(self) -> list[str]:
        return self.list_metrics()

    def evaluate(self, *, model: str = 'hf', model_args: str, tasks, batch_size='auto', device=None, num_fewshot=None, **kwargs) -> dict[str, Any]:
        if not self.available():
            raise ImportError('Install lm-eval: pip install "lm_eval[hf]"')
        from lm_eval import evaluator
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(',') if t.strip()]
        eval_kwargs = dict(model=model, model_args=model_args, tasks=tasks, batch_size=batch_size, device=device)
        if num_fewshot is not None:
            eval_kwargs['num_fewshot'] = num_fewshot
        eval_kwargs.update(kwargs)
        results = evaluator.simple_evaluate(**eval_kwargs)
        return {
            'schema': 'rpipe.metric_benchmark.v1',
            'algorithm': self.name,
            'algorithm_type': self.algorithm_type,
            'mode': self.mode,
            'tasks': tasks,
            'results': results.get('results', results),
        }

    def run(self, **kwargs):
        return self.evaluate(**kwargs)
