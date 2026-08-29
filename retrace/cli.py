"""``python -m retrace <command>`` - one sub-command per pipeline stage.

    prepare              CSV -> knowledge artifacts            (CPU, seconds)
    train                LoRA knowledge injection + gate       (GPU)
    erase   <target>     resolve request + train eraser LoRA   (GPU)
    verify  <group_id>   run the verification harness          (GPU)
    report  <group_id>   render the Erasure Report             (CPU)
    pipeline <target>    erase -> verify -> report in one go   (GPU)
    serve                print the Streamlit launch command

``-v`` / ``-vv`` raise log verbosity and may appear before or after the command.
"""

from __future__ import annotations

import argparse
import logging
import sys

from retrace.exceptions import RetraceError


def _log(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #


def _prepare(a: argparse.Namespace) -> int:
    from retrace.config import KnowledgeConfig
    from retrace.knowledge import build_knowledge_base

    over: dict[str, object] = {}
    if a.csv:
        over["csv_path"] = a.csv
    if a.out:
        over["out_dir"] = a.out
    if a.paraphrases is not None:
        over["paraphrases_per_fact"] = a.paraphrases
    if a.no_strict:
        over["strict"] = False
    res = build_knowledge_base(KnowledgeConfig(**over))
    print(res)
    return 0


def _train(a: argparse.Namespace) -> int:
    from retrace.config import TrainingConfig
    from retrace.training import build_baseline

    over: dict[str, object] = {}
    for src, key in (
        (a.base, "base_model"), (a.knowledge, "knowledge_dir"), (a.out, "out_dir"),
        (a.epochs, "num_epochs"), (a.lora_r, "lora_r"), (a.batch_size, "per_device_batch_size"),
    ):
        if src is not None:
            over[key] = src
    res = build_baseline(TrainingConfig(**over), skip_train=a.skip_train)
    print(res)
    print(f"\nbaseline model: {res.model_dir}")
    return 0 if res.gate_passed else 3


def _erase(a: argparse.Namespace) -> int:
    from retrace.config import ErasureConfig
    from retrace.erasure import erase_entity

    over: dict[str, object] = {}
    if a.max_steps is not None:
        over["max_steps"] = a.max_steps
    if a.lr is not None:
        over["learning_rate"] = a.lr
    if a.npo_beta is not None:
        over["npo_beta"] = a.npo_beta
    res = erase_entity(a.target, ErasureConfig(**over))
    print(res)
    return 0 if res.eraser.converged else 3


def _verify(a: argparse.Namespace) -> int:
    from retrace.verification import run_verification

    report = run_verification(a.group_id)
    s = report["scores"]
    print(f"retrace score (weighted)       {s['retrace_score_weighted']:.3f}")
    print(f"retrace score (multiplicative) {s['retrace_score_multiplicative']:.3f}")
    print(f"forget efficacy                {s['forget_efficacy']:.3f}")
    print(f"retain preservation            {s['retain_preservation']:.3f}")
    print(f"capability preservation        {s['capability_preservation']:.3f}")
    print(f"adversarial resistance         {s['adversarial_resistance']:.3f}")
    return 0


def _report(a: argparse.Namespace) -> int:
    from retrace.reporting import generate_report

    res = generate_report(a.group_id)
    print(f"markdown: {res.markdown_path}")
    if res.html_path:
        print(f"html:     {res.html_path}")
    return 0


def _pipeline(a: argparse.Namespace) -> int:
    from retrace.config import ErasureConfig
    from retrace.erasure import erase_entity
    from retrace.reporting import generate_report
    from retrace.verification import run_verification

    run = erase_entity(a.target, ErasureConfig())
    gid = run.request.target_group_id
    print(run)
    run_verification(gid)
    res = generate_report(gid)
    print(f"\nreport: {res.markdown_path}")
    return 0


def _serve(a: argparse.Namespace) -> int:
    print("Launch the demo with:\n")
    print("    streamlit run retrace/serving/app.py\n")
    print("(needs the 'serve' and 'train' extras). On Colab, tunnel port 8501.")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="count", default=0)

    p = argparse.ArgumentParser(prog="retrace", parents=[common], description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", parents=[common], help="build knowledge artifacts")
    sp.add_argument("--csv"); sp.add_argument("--out")
    sp.add_argument("--paraphrases", type=int)
    sp.add_argument("--no-strict", action="store_true")
    sp.set_defaults(func=_prepare)

    sp = sub.add_parser("train", parents=[common], help="train the baseline model")
    sp.add_argument("--base"); sp.add_argument("--knowledge"); sp.add_argument("--out")
    sp.add_argument("--epochs", type=float)
    sp.add_argument("--lora-r", type=int, dest="lora_r")
    sp.add_argument("--batch-size", type=int, dest="batch_size")
    sp.add_argument("--skip-train", action="store_true")
    sp.set_defaults(func=_train)

    sp = sub.add_parser("erase", parents=[common], help="resolve request + train eraser")
    sp.add_argument("target", help="entity name or group_id")
    sp.add_argument("--max-steps", type=int, dest="max_steps")
    sp.add_argument("--lr", type=float)
    sp.add_argument("--npo-beta", type=float, dest="npo_beta")
    sp.set_defaults(func=_erase)

    sp = sub.add_parser("verify", parents=[common], help="run the verification harness")
    sp.add_argument("group_id")
    sp.set_defaults(func=_verify)

    sp = sub.add_parser("report", parents=[common], help="render the Erasure Report")
    sp.add_argument("group_id")
    sp.set_defaults(func=_report)

    sp = sub.add_parser("pipeline", parents=[common], help="erase -> verify -> report")
    sp.add_argument("target")
    sp.set_defaults(func=_pipeline)

    sp = sub.add_parser("serve", parents=[common], help="how to launch the demo")
    sp.set_defaults(func=_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log(args.verbose)
    try:
        return args.func(args)
    except RetraceError as exc:
        logging.getLogger("retrace").error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(
            f"error: {exc}\n"
            'This command needs the training stack: pip install -e ".[train]"\n'
            "or run it on a GPU host (Colab T4).",
            file=sys.stderr,
        )
        return 4
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
