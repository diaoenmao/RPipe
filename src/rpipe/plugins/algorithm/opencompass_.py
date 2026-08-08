from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('algorithm', 'opencompass')
class OpenCompassMetricProvider:
    """PyPI package: ``opencompass`` — broad LLM benchmark suite (CN/EN).

    Mainstream companion to ``lm_eval`` when you need wider dataset coverage.

    Refs: https://pypi.org/project/opencompass/ · https://github.com/open-compass/opencompass
          https://opencompass.readthedocs.io/
    """

    name = 'opencompass'
    package = 'opencompass'
    mode = 'benchmark'
    kind = 'metric'

    def available(self) -> bool:
        try:
            import opencompass  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return ['mmlu', 'cmmlu', 'ceval', 'gsm8k', 'humaneval', 'bbh']

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        if not self.available():
            raise ImportError('Install package: pip install -U opencompass')
        # OpenCompass is config/CLI driven; expose a thin programmatic hook when possible.
        try:
            from opencompass.cli.main import main as oc_main  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise NotImplementedError(
                'opencompass is installed but programmatic evaluate() needs a config path. '
                'Pass config=... or use the opencompass CLI. Prefer lm_eval for GSM8K quick path.'
            ) from exc
        config = kwargs.get('config')
        if not config:
            raise ValueError('opencompass.evaluate requires config= path to an OpenCompass config')
        # Delegate to CLI entry (side-effectful); callers should prefer subprocess for full runs.
        return {
            'schema': 'rpipe.metric_benchmark.v1',
            'provider': self.name,
            'package': self.package,
            'mode': self.mode,
            'config': config,
            'note': 'Use OpenCompass CLI for full multi-dataset runs; this returns a handoff payload.',
            'cli': f'opencompass {config}',
        }
