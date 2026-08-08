from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('algorithm', 'lm_eval')
class LMEvalMetricProvider:
    """PyPI package: ``lm_eval`` (EleutherAI lm-evaluation-harness).

    LLM benchmark metrics (GSM8K, MMLU, …). Same algorithm registry as torchmetrics;
    ``mode='benchmark'`` instead of online batch metrics.

    Refs: https://pypi.org/project/lm_eval/ · https://github.com/EleutherAI/lm-evaluation-harness
    """

    name = 'lm_eval'
    package = 'lm_eval'
    mode = 'benchmark'
    kind = 'metric'

    def available(self) -> bool:
        try:
            import lm_eval  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return self.list_tasks()

    def list_tasks(self) -> list[str]:
        return [
            'gsm8k', 'gsm8k_cot', 'mmlu', 'hellaswag', 'arc_challenge',
            'truthfulqa_mc2', 'winogrande', 'bbh', 'humaneval', 'ifeval',
        ]

    def evaluate(
        self,
        *,
        model: str = 'hf',
        model_args: str,
        tasks: list[str] | str,
        batch_size: str | int = 'auto',
        device: str | None = None,
        num_fewshot: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if not self.available():
            raise ImportError('Install lm-eval: pip install "lm_eval[hf]"')
        from lm_eval import evaluator

        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(',') if t.strip()]
        eval_kwargs = dict(
            model=model,
            model_args=model_args,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
        )
        if num_fewshot is not None:
            eval_kwargs['num_fewshot'] = num_fewshot
        eval_kwargs.update(kwargs)
        results = evaluator.simple_evaluate(**eval_kwargs)
        return {
            'schema': 'rpipe.metric_benchmark.v1',
            'provider': self.name,
            'mode': self.mode,
            'tasks': tasks,
            'results': results.get('results', results),
            'config': results.get('config'),
        }

    # alias used by older call sites
    def run(self, **kwargs):
        return self.evaluate(**kwargs)
