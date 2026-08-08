from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('data', 'datasets')
class DatasetsDataProvider:
    """PyPI package: ``datasets`` (Hugging Face Datasets).

    Refs: https://pypi.org/project/datasets/ · https://huggingface.co/docs/datasets
    """

    name = 'datasets'
    package = 'datasets'

    def available(self) -> bool:
        try:
            import datasets  # noqa: F401
            return True
        except ImportError:
            return False

    def list_datasets(self) -> list[str]:
        return [
            'openai/gsm8k',
            'cais/mmlu',
            'glue',
            'imdb',
            'squad',
        ]

    def build(
        self,
        data_name: str,
        *,
        process: bool = False,
        verbose: bool = True,
        split_map: dict[str, str] | None = None,
        text_field: str | None = None,
        label_field: str | None = None,
        subset: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if not self.available():
            raise ImportError('Install package: pip install datasets')
        from datasets import load_dataset
        from torch.utils.data import Dataset

        if verbose:
            print(f'[datasets] loading {data_name}' + (f' ({subset})' if subset else ''))

        aliases = {
            'GSM8K': ('openai/gsm8k', 'main'),
            'gsm8k': ('openai/gsm8k', 'main'),
        }
        path, config = data_name, subset
        if data_name in aliases:
            path, config = aliases[data_name]

        ds = load_dataset(path, config, **{k: v for k, v in kwargs.items() if k not in ('data_root',)})
        split_map = split_map or {'train': 'train', 'test': 'test'}
        if 'test' not in ds and 'validation' in ds:
            split_map = {**split_map, 'test': 'validation'}

        text_field = text_field or _guess_text_field(ds[list(ds.keys())[0]].column_names)
        label_field = label_field or _guess_label_field(ds[list(ds.keys())[0]].column_names)

        class _HFTorch(Dataset):
            def __init__(self, table, split_name):
                self.table = table
                self.split_name = split_name
                self.data_size = None
                self.target_size = None

            def __len__(self):
                return len(self.table)

            def __getitem__(self, idx):
                row = self.table[idx]
                item = {'id': idx, 'data': row.get(text_field), 'raw': row}
                if label_field and label_field in row:
                    item['target'] = row[label_field]
                return item

        out = {}
        for key, split in split_map.items():
            if split not in ds:
                continue
            out[key] = _HFTorch(ds[split], split)
        if 'train' not in out:
            raise ValueError(f'datasets {data_name} missing train split; have {list(ds.keys())}')
        if 'test' not in out:
            out['test'] = out['train']
        if verbose:
            print('[datasets] ready', {k: len(v) for k, v in out.items()})
        return out


def _guess_text_field(columns: list[str]) -> str:
    for c in ('question', 'text', 'sentence', 'content', 'prompt', 'input'):
        if c in columns:
            return c
    return columns[0]


def _guess_label_field(columns: list[str]) -> str | None:
    for c in ('answer', 'label', 'target', 'output'):
        if c in columns:
            return c
    return None
