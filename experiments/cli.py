"""Experiment-layer CLI (outside the rpipe package)."""

from __future__ import annotations

import argparse


def build_parser():
    from experiments import STAGES
    parser = argparse.ArgumentParser(
        prog='rpipe-run',
        description='Run research suites: prepare → train → test → process → artifacts',
    )
    parser.add_argument('--suite', default='smoke', help='Suite name from configs/suites/default.yaml')
    parser.add_argument('--stages', default=','.join(STAGES), help='Comma-separated stages')
    parser.add_argument('--device', default=None, help='cpu | cuda (default: auto)')
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--force-prepare', action='store_true')
    parser.add_argument('--list-suites', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    from experiments import ResearchPipeline, list_suites

    if args.list_suites:
        for name in list_suites():
            print(name)
        return 0

    device = args.device
    if device is None:
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

    result = ResearchPipeline(
        suite_name=args.suite,
        device=device,
        stages=args.stages,
        force_prepare=args.force_prepare,
        output_root=args.output_root,
    ).run()
    print('[pipeline] done:', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
