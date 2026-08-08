"""Generate Markdown test report from persisted JSONL results (研讨纪要 §七)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = REPO_ROOT / 'output' / 'test_results' / 'results.jsonl'
DEFAULT_META = REPO_ROOT / 'output' / 'test_results' / 'run_meta.json'
DEFAULT_OUT = REPO_ROOT / 'output' / 'test_results' / 'TEST_REPORT.md'


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _pct(n: int, d: int) -> str:
    return f'{100.0 * n / d:.1f}%' if d else 'n/a'


def render(rows: list[dict], meta: dict | None = None) -> str:
    meta = meta or {}
    total = len(rows)
    passed = sum(1 for r in rows if r['outcome'] == 'passed')
    failed = sum(1 for r in rows if r['outcome'] == 'failed')
    skipped = sum(1 for r in rows if r['outcome'] == 'skipped')
    errors = sum(1 for r in rows if r['outcome'] not in ('passed', 'failed', 'skipped'))

    lines: list[str] = []
    lines.append('# RPipe 测试报告')
    lines.append('')
    lines.append(f'- 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'- 规范依据：研讨纪要 测试规范（unit / integration / e2e × location / content / physical × p1–p3）')
    lines.append(f'- CUDA：{meta.get("cuda_available", "unknown")}（{meta.get("device_name", "?")}）')
    lines.append(f'- torch：{meta.get("torch_version", "?")}')
    lines.append('')
    lines.append('## 1. 总览')
    lines.append('')
    lines.append('| 指标 | 数量 |')
    lines.append('|------|------|')
    lines.append(f'| Total | {total} |')
    lines.append(f'| Pass | {passed} ({_pct(passed, total)}) |')
    lines.append(f'| Fail | {failed} ({_pct(failed, total)}) |')
    lines.append(f'| Skip | {skipped} |')
    if errors:
        lines.append(f'| Other | {errors} |')
    lines.append('')

    def section(title: str, key: str):
        lines.append(f'## {title}')
        lines.append('')
        lines.append('| 维度 | Pass | Fail | Skip | Total | Pass率 |')
        lines.append('|------|------|------|------|-------|--------|')
        buckets: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            val = r.get(key) or 'untagged'
            if isinstance(val, list):
                keys = val or ['untagged']
            else:
                keys = [val]
            for k in keys:
                buckets[k][r['outcome']] += 1
                buckets[k]['total'] += 1
        for name in sorted(buckets):
            c = buckets[name]
            t = c['total']
            lines.append(
                f'| `{name}` | {c["passed"]} | {c["failed"]} | {c["skipped"]} | {t} | {_pct(c["passed"], t - c["skipped"])} |'
            )
        lines.append('')

    section('2. 按优先级（Priority）', 'priority')
    section('3. 按测试层级（Level）', 'level')
    section('4. 按测试类型（Type）', 'type')
    section('5. 按架构层（Architecture Layer）', 'layers')
    section('6. 按模块 Tag（module_*）', 'modules')
    section('7. 按功能 Tag（feature_*）', 'features')

    lines.append('## 8. 集成 / 端到端')
    lines.append('')
    ie = [r for r in rows if r.get('level') in ('integration', 'e2e')]
    if not ie:
        lines.append('_（本轮无 integration/e2e 结果）_')
        lines.append('')
    else:
        lines.append('| Level | 测试 | 结果 | 时长(s) | markers |')
        lines.append('|-------|------|------|---------|---------|')
        for r in ie:
            lines.append(
                f'| `{r.get("level")}` | `{r["name"]}` | **{r["outcome"]}** | '
                f'{r.get("duration_s") or "-"} | {", ".join(r.get("markers") or [])} |'
            )
        lines.append('')

    fails = [r for r in rows if r['outcome'] == 'failed']
    lines.append('## 9. 失败用例与定位')
    lines.append('')
    if not fails:
        lines.append('无失败。')
        lines.append('')
    else:
        for r in fails:
            lines.append(f'### `{r["nodeid"]}`')
            lines.append('')
            lines.append(f'- 路径：`{r["path"]}`')
            lines.append(f'- Tags：{", ".join(r.get("markers") or [])}')
            lines.append(f'- 建议：从对应 `src/` 镜像位置检查契约与依赖（见失败摘要）')
            lines.append('')
            lines.append('```')
            lines.append((r.get('longrepr') or '')[:4000])
            lines.append('```')
            lines.append('')

    lines.append('## 10. 后续建议')
    lines.append('')
    if failed:
        lines.append('- 优先修复 **p1** 失败，再扩大到 p2/p3。')
        lines.append('- 按失败用例的 `*_layer` / `module_*` Tag 缩小排查范围。')
    else:
        lines.append('- 本轮全部通过；可将 `external`/`slow` 范围扩大到可选第三方冒烟。')
    if not meta.get('cuda_available'):
        lines.append('- 当前环境 **无 CUDA**；GPU physical / e2e 已按 CPU 回退或 skip。有 GPU 时重跑 `pytest -m "gpu or e2e"`。')
    lines.append('- 结果落盘：`output/test_results/results.jsonl`；可重复解析生成本报告。')
    lines.append('')
    lines.append('## 附录：全部用例')
    lines.append('')
    lines.append('| Outcome | Priority | Level | Type | Name |')
    lines.append('|---------|----------|-------|------|------|')
    for r in rows:
        lines.append(
            f'| {r["outcome"]} | {r.get("priority")} | {r.get("level")} | {r.get("type")} | `{r["name"]}` |'
        )
    lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, default=DEFAULT_JSONL)
    ap.add_argument('--meta', type=Path, default=DEFAULT_META)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = load_results(args.jsonl)
    meta = json.loads(args.meta.read_text(encoding='utf-8')) if args.meta.exists() else {}
    text = render(rows, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf-8')
    print(f'Wrote {args.out} ({len(rows)} tests)')


if __name__ == '__main__':
    main()
