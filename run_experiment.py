#!/usr/bin/env python
"""
CLI entry point for running experiments.

Usage examples:
    python run_experiment.py --experiment-type base --method add
    python run_experiment.py --experiment-type exception_handling --method generate_bpmn --sample 25
    python run_experiment.py --experiment-type exception_handling_db_error --method add --sample 25 --model gpt-4o
    python run_experiment.py --evaluate-only --experiment-type base --method add
    python run_experiment.py --setup-db
    python calculate_results.py  # aggregate results (separate script)
"""
import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.ollama")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments")

    # Mode flags
    parser.add_argument("--setup-db", action="store_true", help="Create/recreate the database")
    parser.add_argument("--setup-all-dbs", action="store_true", help="Create both clean template databases (database_clean.db + database_clean_typo.db)")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate existing sessions")
    # Experiment config
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="base",
        choices=["base", "exception_handling", "exception_handling_db_error"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["add", "generate_bpmn", "classic"],
        help="Rule adaptation method (default: all)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample size (for exception handling experiments)")
    parser.add_argument("--test", action="store_true", help="Test mode: 1 adaptation (base_rule), 1 tour (J09A), 1 seed (42)")
    parser.add_argument("--model", type=str, default=None, help="Model name (or set LLM_MODEL env var). Non-gpt/claude models auto-use Ollama.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL (or set LLM_BASE_URL env var)")
    parser.add_argument("--seed", type=int, default=None, help="Run with a single seed (e.g. --seed 42)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for evaluation results")

    return parser.parse_args()


def cmd_setup_db(args):
    from db.setup import create_database, generate_master_csvs

    use_typo = args.experiment_type in ("exception_handling", "exception_handling_db_error")
    generate_master_csvs()
    create_database(use_schema_typo=use_typo)
    if use_typo:
        print("Note: decision table created as 'decision_talbe' (intentional typo for exception handling)")



def cmd_evaluate(args):
    from config.server_config import ExperimentType
    from experiment.evaluator import Evaluator
    from experiment.runner import ExperimentRunner

    experiment_type = ExperimentType(args.experiment_type)
    methods = [args.method] if args.method else ["add", "generate_bpmn"]

    config = _build_config(args, methods)
    runner = ExperimentRunner(config)
    runs = runner.generate_runs()

    evaluator = Evaluator(sessions_folder=config.sessions_folder)
    df_results = evaluator.evaluate_all(runs, experiment_type, model_name=config.model_config.model)

    if args.output:
        output_path = args.output
    else:
        os.makedirs("experiment_results", exist_ok=True)
        model_dir = config.model_config.model.replace(":", "_").replace("/", "_")
        method_str = args.method if args.method else "all"
        suffix = experiment_type.session_suffix if experiment_type.session_suffix else ""
        output_path = f"experiment_results/process_adaptation_results_{model_dir}_{method_str}{suffix}.csv"

    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Overall success rate: {df_results['all_correct'].mean():.2%}")


def _build_config(args, methods):
    from config.experiment_config import ExperimentConfig
    from config.model_config import ModelConfig
    from config.server_config import ExperimentType

    experiment_type = ExperimentType(args.experiment_type)

    model_kwargs = {"temperature": 0}
    if args.model:
        model_kwargs["model"] = args.model
    if args.base_url:
        model_kwargs["base_url"] = args.base_url
    model_config = ModelConfig(**model_kwargs)

    # Sanitize model name for directory use
    model_dir = model_config.model.replace(":", "_").replace("/", "_")
    sessions_folder = f"sessions/{model_dir}"

    config_kwargs = {
        "experiment_type": experiment_type,
        "model_config": model_config,
        "rule_adaptation_methods": methods,
        "sample_size": args.sample,
        "sessions_folder": sessions_folder,
    }

    if args.seed is not None:
        config_kwargs["seeds"] = [args.seed]
        print(f"[SINGLE SEED] Running with seed {args.seed}")

    if args.test:
        config_kwargs["process_adaptations"] = ["base_rule"]
        config_kwargs["tours"] = ["J09A"]
        config_kwargs["seeds"] = [42]
        print("[TEST MODE] 1 adaptation (base_rule), 1 tour (J09A), 1 seed (42)")

    return ExperimentConfig(**config_kwargs)


async def cmd_run(args):
    from experiment.runner import ExperimentRunner

    methods = [args.method] if args.method else ["add", "generate_bpmn"]
    config = _build_config(args, methods)
    config.model_config.validate_model()

    runner = ExperimentRunner(config)
    await runner.run_all()


def main():
    args = parse_args()

    if args.setup_all_dbs:
        from db.setup import create_all_clean_databases
        create_all_clean_databases()
        return

    if args.setup_db:
        cmd_setup_db(args)
        return

    if args.evaluate_only:
        cmd_evaluate(args)
        return

    # Default: run experiment
    asyncio.run(cmd_run(args))


if __name__ == "__main__":
    main()
