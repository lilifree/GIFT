#!/usr/bin/env python3
"""Unit tests for GIFT piecewise scoring, scope, and token annealing."""

from __future__ import annotations

import inspect
import math
import unittest

import torch

from core_runtime import patch as core
from gift_runtime.patch import (
    build_gift_masks,
    sparse_tail_gradient_geometry,
)


def geometry(
    advantages: torch.Tensor,
    *,
    old_shift: float = 0.0,
    teacher_topk_probability: float = 0.1,
) -> dict:
    # Student distribution: sampled action 0, teacher-top1 index 1, tail index 2.
    student = torch.tensor([0.3, 0.2, 0.5], dtype=torch.float64)
    length = int(advantages.numel())
    student_topk = student[1].log().repeat(1, length, 1)
    teacher_topk = torch.tensor(
        teacher_topk_probability, dtype=torch.float64
    ).log().repeat(1, length, 1)
    student_action = student[0].log().repeat(1, length)
    return sparse_tail_gradient_geometry(
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
        topk_indices=torch.ones((1, length, 1), dtype=torch.long),
        actions=torch.zeros((1, length), dtype=torch.long),
        student_action_log_probs=student_action,
        student_sum_pi_squared=student.square().sum().repeat(1, length),
        old_action_log_probs=student_action - float(old_shift),
        advantages=advantages.reshape(1, length),
        response_mask=torch.ones((1, length), dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        dual_clip_ratio=3.0,
        score_mode="piecewise_conflict_residual",
    )


def record(uids: list[str], scores: list[list[float]]) -> dict:
    score = torch.tensor(scores, dtype=torch.float32)
    shape = score.shape
    valid = torch.ones(shape, dtype=torch.bool)
    false = torch.zeros(shape, dtype=torch.bool)
    return {
        "uids": uids,
        "score": score,
        "eligible": valid.clone(),
        "valid": valid,
        "active": valid.clone(),
        "cosine": torch.full(shape, 0.5),
        "residual": score.abs(),
        "parallel": torch.where(score < 0, score, torch.full(shape, 0.25)),
        "conflict": score < 0,
        "neutral_grpo": false.clone(),
        "direction_defined": valid.clone(),
        "teacher_zero": false.clone(),
        "r_norm": torch.ones(shape),
        "d_norm": torch.ones(shape),
        "ratio": torch.ones(shape),
        "advantage": torch.ones(shape),
        "action_in_topk": false.clone(),
    }


class TestPiecewiseGeometry(unittest.TestCase):
    def test_sign_selects_branch_and_advantage_magnitude_does_not_rank(self):
        result = geometry(torch.tensor([1.0, 3.0, -1.0, -3.0]))
        score = result["score"][0]
        parallel = result["parallel"][0]
        residual = result["residual"][0]
        self.assertGreater(float(parallel[0]), 0.0)
        self.assertLess(float(parallel[2]), 0.0)
        self.assertAlmostEqual(float(score[0]), float(residual[0]), places=6)
        self.assertAlmostEqual(float(score[1]), float(score[0]), places=6)
        self.assertAlmostEqual(float(score[2]), float(parallel[2]), places=6)
        self.assertAlmostEqual(float(score[3]), float(score[2]), places=6)
        self.assertGreater(float(score[0]), float(score[2]))

    def test_parallel_and_residual_match_true_logit_autograd(self):
        advantage = 2.0
        result = geometry(torch.tensor([advantage], dtype=torch.float64))
        logits = torch.log(
            torch.tensor([0.3, 0.2, 0.5], dtype=torch.float64)
        ).requires_grad_()
        log_probs = torch.log_softmax(logits, dim=-1)
        ratio = torch.exp(log_probs[0] - log_probs[0].detach())
        pg_loss = -advantage * ratio
        q_top, q_tail = (
            torch.tensor(0.1, dtype=torch.float64),
            torch.tensor(0.9, dtype=torch.float64),
        )
        kl_loss = q_top * (q_top.log() - log_probs[1]) + q_tail * (
            q_tail.log() - torch.logsumexp(log_probs[[0, 2]], dim=0)
        )
        r = torch.autograd.grad(pg_loss, logits, retain_graph=True)[0]
        d = torch.autograd.grad(kl_loss, logits)[0]
        a = torch.dot(d, r / torch.linalg.vector_norm(r))
        b = torch.linalg.vector_norm(d - a * r / torch.linalg.vector_norm(r))
        self.assertAlmostEqual(float(result["parallel"].item()), float(a), places=5)
        self.assertAlmostEqual(float(result["residual"].item()), float(b), places=5)

    def test_zero_grpo_is_neutral_but_zero_teacher_is_excluded(self):
        neutral = geometry(torch.tensor([0.0]))
        self.assertTrue(bool(neutral["eligible"].item()))
        self.assertTrue(bool(neutral["neutral_grpo"].item()))
        self.assertEqual(float(neutral["score"].item()), 0.0)

        zero_teacher = geometry(
            torch.tensor([1.0]), teacher_topk_probability=0.2
        )
        self.assertFalse(bool(zero_teacher["eligible"].item()))
        self.assertTrue(bool(zero_teacher["teacher_zero"].item()))

    def test_clipped_grpo_uses_neutral_fallback(self):
        clipped = geometry(torch.tensor([1.0]), old_shift=math.log(1.3))
        self.assertFalse(bool(clipped["active"].item()))
        self.assertTrue(bool(clipped["neutral_grpo"].item()))
        self.assertEqual(float(clipped["score"].item()), 0.0)

    def test_selector_interface_has_no_entropy(self):
        self.assertNotIn(
            "entropy", inspect.signature(sparse_tail_gradient_geometry).parameters
        )


class TestSelectionScopes(unittest.TestCase):
    def setUp(self):
        # One real microbatch containing two complete prompt groups, two
        # rollouts per group, and two response tokens per rollout.
        self.records = [
            record(
                ["a", "a", "b", "b"],
                [[1.0, 2.0], [3.0, 4.0], [9.0, 8.0], [7.0, 6.0]],
            )
        ]

    def test_group_has_independent_per_prompt_budgets(self):
        queue, diagnostics = build_gift_masks(
            self.records,
            0.5,
            group_size=2,
            selection_scope="prompt_group",
        )
        self.assertEqual(queue[0]["mask"].sum(dim=1).tolist(), [0.0, 2.0, 2.0, 0.0])
        self.assertEqual(int(diagnostics["selected"]), 4)
        self.assertEqual(int(diagnostics["scope_count"]), 2)

    def test_microbatch_can_transfer_budget_between_prompts(self):
        queue, diagnostics = build_gift_masks(
            self.records,
            0.5,
            group_size=2,
            selection_scope="micro_batch",
        )
        self.assertEqual(queue[0]["mask"].sum(dim=1).tolist(), [0.0, 0.0, 2.0, 2.0])
        self.assertEqual(int(diagnostics["selected"]), 4)
        self.assertEqual(int(diagnostics["scope_count"]), 1)

    def test_piecewise_order_removes_strong_conflict_first(self):
        data = record(
            ["a", "a"],
            [[-0.9, -0.1, 0.0], [0.2, 0.8, 0.4]],
        )
        queue, _ = build_gift_masks(
            [data], 0.5, group_size=2, selection_scope="prompt_group"
        )
        chosen = queue[0]["mask"].bool()
        chosen_values = data["score"][chosen].sort().values.tolist()
        for actual, expected in zip(chosen_values, [0.2, 0.4, 0.8], strict=True):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_full_budget_keeps_every_eligible_token(self):
        data = record(["a", "a"], [[-1.0, 0.0], [1.0, 2.0]])
        data["eligible"][0, 1] = False
        data["teacher_zero"][0, 1] = True
        queue, diagnostics = build_gift_masks(
            [data], 1.0, group_size=2, selection_scope="prompt_group"
        )
        self.assertEqual(int(queue[0]["selected"]), 3)
        self.assertEqual(int(diagnostics["budget"]), 4)
        self.assertEqual(float(queue[0]["mask"][0, 1]), 0.0)


class TestSchedule(unittest.TestCase):
    config = {
        "gift_hold_steps": 10,
        "gift_exit_step": 78,
        "gift_initial_retention": 1.0,
        "opsd_grpo_lambda": 1.0,
    }

    def test_exact_full_start_linear_exit_schedule(self):
        ratio = core.coverage_ratio
        expected = {
            1: 1.0,
            10: 1.0,
            11: 67.0 / 68.0,
            44: 0.5,
            77: 1.0 / 68.0,
            78: 0.0,
            96: 0.0,
        }
        for step, value in expected.items():
            self.assertAlmostEqual(ratio(self.config, step), value, places=12)
        values = [ratio(self.config, step) for step in range(1, 97)]
        self.assertTrue(all(b <= a for a, b in zip(values, values[1:])))

    def test_lambda_never_anneals(self):
        self.assertTrue(
            all(core.opsd_lambda(self.config, step) == 1.0 for step in range(1, 97))
        )

    def test_schedule_follows_configured_steps(self):
        # Nothing about the paper's 10/78/96 is baked in: any 0 < H < T_exit works.
        config = dict(self.config, gift_hold_steps=5, gift_exit_step=45)
        ratio = core.coverage_ratio
        expected = {1: 1.0, 5: 1.0, 6: 39.0 / 40.0, 25: 0.5, 44: 1.0 / 40.0, 45: 0.0, 200: 0.0}
        for step, value in expected.items():
            self.assertAlmostEqual(ratio(config, step), value, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
