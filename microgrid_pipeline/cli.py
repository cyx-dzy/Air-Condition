from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="光储荷微电网预测与风险调度 Python 复现")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="使用模拟数据运行完整流程")
    demo_parser.add_argument("--output", default="outputs", help="输出目录")

    run_parser = subparsers.add_parser("run", help="使用真实 CSV 运行完整流程")
    run_parser.add_argument("--data", required=True, help="真实 CSV 路径")
    run_parser.add_argument("--output", default="outputs", help="输出目录")

    args = parser.parse_args()
    if args.command == "demo":
        artifacts = run_pipeline(data_path=None, output_dir=args.output)
    else:
        artifacts = run_pipeline(data_path=args.data, output_dir=args.output)

    print("运行完成，已生成：")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")

