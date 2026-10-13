#!/usr/bin/env python3
"""CPU-only contracts for the 4B/1.7B GIFT math scale experiments."""

from __future__ import annotations

from pathlib import Path
import unittest

from core_runtime import patch as core
from gift_runtime.patch import GiftSettings
from verify_gift_model_scale_math import SPECS, resolve_path, validate_config


HERE = Path(__file__).resolve().parent


class GiftModelScaleMathConfigTest(unittest.TestCase):
    def test_both_configs_are_exact_scale_reproductions(self) -> None:
        outputs: set[str] = set()
        for spec in SPECS.values():
            raw = validate_config(HERE / spec.config_name)
            self.assertEqual(raw["gift_selection_scope"], "micro_batch")
            self.assertFalse(raw["val_before_train"])
            self.assertEqual(raw["save_steps"], 10)
            self.assertEqual(raw["eval_steps"], 10)
            self.assertEqual(resolve_path(raw["model_path"]).name, spec.model_name)
            outputs.add(raw["output_dir"])
        self.assertEqual(len(outputs), 2)

    def test_token_only_schedule_matches_the_8b_contract(self) -> None:
        for spec in SPECS.values():
            raw = validate_config(HERE / spec.config_name)
            settings = GiftSettings.from_mapping(raw)
            total = settings.total_training_steps
            hold = settings.hold_steps
            exit_step = settings.exit_step
            ratios = [
                core.coverage_ratio(raw, step)
                for step in range(1, total + 1)
            ]
            lambdas = [core.opsd_lambda(raw, step) for step in range(1, total + 1)]
            self.assertEqual(ratios[:hold], [1.0] * hold)
            midpoint = (hold + exit_step) / 2
            if midpoint.is_integer():
                self.assertAlmostEqual(ratios[int(midpoint) - 1], 0.5)
            self.assertEqual(ratios[exit_step - 1 :], [0.0] * (total - exit_step + 1))
            self.assertEqual(lambdas, [1.0] * total)


if __name__ == "__main__":
    unittest.main()
