from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .config import load_config
from .pipeline import ResearchPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-ashare", description="PIT、向量化的 A 股机器学习研究平台"
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML 配置路径")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="检查运行环境")
    update = commands.add_parser("update", help="增量更新 TickFlow 行情")
    update.add_argument("--start")
    update.add_argument("--end")
    build = commands.add_parser("build", help="构建特征和标签")
    build.add_argument("--year", type=int)
    for name in ("train", "walk-forward"):
        command = commands.add_parser(name, help=f"运行 {name}")
        command.add_argument("--model", choices=("lightgbm", "ridge", "catboost-gpu"))
    backtest = commands.add_parser("backtest", help="运行成交级回测")
    backtest.add_argument("--predictions")
    backtest.add_argument("--capital", type=float)
    daily = commands.add_parser("daily", help="更新并生成日报")
    daily.add_argument("--skip-update", action="store_true")
    commands.add_parser("weekly", help="生成周报")
    commands.add_parser("gui", help="启动 Windows 桌面界面")
    return parser


def _print(result: Any) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config_path = Path(args.config)
    config = load_config(config_path)
    if args.command == "gui":
        from .gui import launch_gui

        launch_gui(config_path)
        return 0
    pipeline = ResearchPipeline(config)
    if args.command == "doctor":
        result = pipeline.doctor()
    elif args.command == "update":
        result = pipeline.update(args.start, args.end)
    elif args.command == "build":
        result = pipeline.build(args.year)
    elif args.command == "train":
        result = pipeline.train(args.model)
    elif args.command == "walk-forward":
        result = pipeline.walk_forward(args.model)
    elif args.command == "backtest":
        result = pipeline.backtest(args.predictions, args.capital)
    elif args.command == "daily":
        result = pipeline.daily(args.skip_update)
    elif args.command == "weekly":
        result = pipeline.weekly()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
