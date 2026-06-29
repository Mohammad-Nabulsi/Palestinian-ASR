#!/usr/bin/env python3
"""Create a single-model non-smoke notebook copy for the Layla Omni run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "train_omni_300m_layla_non_smoke.ipynb"


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Omni ASR 300M on Layla — non-smoke notebook copy

            This notebook is configured for a single Layla-backed Omni run.

            It uses:

            - model: `Omni ASR 300M`
            - train: `data/layla_manifests/train.jsonl`
            - val: `data/layla_manifests/val.jsonl`
            - test: `data/layla_manifests/test.jsonl`
            - mode: `smoke_mode = false`

            Current behavior:

            - the notebook now runs end-to-end
            - the Omni non-smoke path currently uses a deterministic placeholder backend
            - that means it will prepare data, write predictions, evaluate them, write checkpoints, and compare base vs tuned outputs
            - it does **not** yet perform real Omni fine-tuning
            """
        ),
        markdown_cell("## 1. Imports"),
        code_cell(
            """
            from pathlib import Path
            import sys

            cwd = Path.cwd().resolve()
            candidate_roots = [
                cwd,
                cwd / "asr_milestone7",
                cwd.parent,
            ]
            project_root = None
            for candidate in candidate_roots:
                if (candidate / "asr_pipeline").exists() and (candidate / "configs").exists():
                    project_root = candidate
                    break

            if project_root is None:
                raise RuntimeError(f"Could not locate asr_milestone7 project root from cwd={cwd}")

            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from asr_pipeline.config import load_config, save_resolved_config_json
            from asr_pipeline.registry import get_model_family, create_adapter
            from asr_pipeline.data import prepare_data_and_collator
            from asr_pipeline.predict import predict
            from asr_pipeline.evaluate import compare_base_vs_tuned, evaluate_predictions
            from asr_pipeline.train import train
            from asr_pipeline.utils.hashing import config_hash
            from asr_pipeline.utils.io import ensure_dir

            print(f"Current working directory: {cwd}")
            print(f"Resolved project root: {project_root}")
            """
        ),
        markdown_cell("## 2. Config setup"),
        code_cell(
            """
            config_path = project_root / "configs" / "omni_300m_layla_3shards.yaml"
            config = load_config(config_path)

            config = type(config)(
                model_name=config.model_name,
                train_path=str((project_root / config.train_path).resolve()),
                val_path=str((project_root / config.val_path).resolve()),
                test_path=str((project_root / config.test_path).resolve()),
                output_dir=str((project_root / config.output_dir).resolve()),
                run_name=config.run_name,
                smoke_mode=False,
                wandb_mode=config.wandb_mode,
                learning_rate=config.learning_rate,
                max_epochs=config.max_epochs,
                early_stopping_patience=config.early_stopping_patience,
                seed=config.seed,
                local_model_cache_dir=str((project_root / config.local_model_cache_dir).resolve()),
            )

            reports_dir = ensure_dir(Path(config.output_dir) / "reports")
            resolved_config_path = reports_dir / f"{config.run_name}_resolved_config.json"
            save_resolved_config_json(config, resolved_config_path)

            print(f"Loaded config from: {config_path}")
            print(f"Resolved config saved to: {resolved_config_path}")
            print(f"Config hash: {config_hash(config)}")
            print(f"Model: {config.model_name}")
            print(f"Smoke mode: {config.smoke_mode}")
            print(f"Train path: {config.train_path}")
            print(f"Val path: {config.val_path}")
            print(f"Test path: {config.test_path}")
            print(f"Output dir: {config.output_dir}")
            """
        ),
        markdown_cell("## 3. Single-model setup"),
        code_cell(
            """
            model_name = config.model_name
            family = get_model_family(model_name)
            adapter = create_adapter(config)
            split_paths = {
                "train": config.train_path,
                "val": config.val_path,
                "test": config.test_path,
            }

            print(f"{model_name} -> {family}")
            print(f"Adapter: {adapter.__class__.__name__}")
            print(adapter.summary())
            """
        ),
        markdown_cell("## 4. Prepare data"),
        code_cell(
            """
            prepared = prepare_data_and_collator(config, adapter, split_paths)
            collator = prepared["collator"]

            print("Prepared split sizes:")
            for split in ("train", "val", "test"):
                print(f"  {split}: {len(prepared['prepared'][split])}")
            print(f"Collator: {collator.__class__.__name__}")
            print(f"Preparation cache path: {prepared['cache_dir']}")
            print(f"Preparation cache status: {prepared['metadata']['cache_status']}")
            """
        ),
        markdown_cell("## 5. Base prediction and evaluation"),
        code_cell(
            """
            base_pred = predict(config, adapter, prepared, split="test")
            base_eval = evaluate_predictions(config, base_pred["metadata"]["path"])

            print("Base prediction path:", base_pred["metadata"]["path"])
            print("Base prediction cache loaded:", base_pred["metadata"]["loaded_from_cache"])
            print("Base metrics path:", base_eval["metadata"]["metrics_path"])
            print("Base WER:", base_eval["metrics"]["wer"])
            """
        ),
        markdown_cell("## 6. Training"),
        code_cell(
            """
            train_result = train(config, adapter, prepared, collator)
            train_meta = train_result["metadata"]

            print("Training status:", train_meta["train_status"])
            print("Training mode:", train_result["training"]["training_mode"])
            print("Best checkpoint:", train_meta["best_checkpoint_path"])
            print("Training metrics path:", train_meta["training_metrics_path"])
            """
        ),
        markdown_cell("## 7. Tuned prediction, evaluation, and comparison"),
        code_cell(
            """
            tuned_pred = predict(
                config,
                adapter,
                prepared,
                split="test",
                tuned_adapter_path=train_meta["best_checkpoint_path"],
            )
            tuned_eval = evaluate_predictions(config, tuned_pred["metadata"]["path"])
            comparison = compare_base_vs_tuned(
                base_eval["metadata"]["metrics_path"],
                tuned_eval["metadata"]["metrics_path"],
            )

            print("Tuned prediction path:", tuned_pred["metadata"]["path"])
            print("Tuned metrics path:", tuned_eval["metadata"]["metrics_path"])
            print("Tuned WER:", tuned_eval["metrics"]["wer"])
            print("Comparison JSON:", comparison["metadata"]["comparison_json_path"])
            print("Comparison Markdown:", comparison["metadata"]["comparison_markdown_path"])
            print("WER improvement:", comparison["comparison"]["metrics"]["wer"]["absolute_improvement"])
            """
        ),
        markdown_cell("## 8. Summary"),
        code_cell(
            """
            summary = {
                "model_name": config.model_name,
                "run_name": config.run_name,
                "output_dir": config.output_dir,
                "base_prediction": base_pred["metadata"]["path"],
                "base_metrics": base_eval["metadata"]["metrics_path"],
                "best_checkpoint": train_meta["best_checkpoint_path"],
                "tuned_prediction": tuned_pred["metadata"]["path"],
                "tuned_metrics": tuned_eval["metadata"]["metrics_path"],
                "comparison_json": comparison["metadata"]["comparison_json_path"],
            }
            print(summary)
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
