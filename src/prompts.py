"""Prompt templates for the legacy trainer path."""

from src.opsd_format import TEACHER_TRANSITION_PROMPT

_BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{{}}."
_THINKING_BOXED_INSTRUCTION = (
    "Please first think step by step inside <think>...</think> tags, "
    "and then put your final answer within \\boxed{{}}."
)


def _instruction(thinking: bool = False) -> str:
    return _THINKING_BOXED_INSTRUCTION if thinking else _BOXED_INSTRUCTION


TEACHER_PROMPT_TEMPLATE_SOLUTION_ANSWER = (
    "Problem: {problem}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n"
    "{solution}\n\n"
    "Correct final answer: {answer}\n"
    "=== Reference Solution End ==="
    f"{TEACHER_TRANSITION_PROMPT}\n"
    f"{_BOXED_INSTRUCTION}"
)

TEACHER_PROMPT_TEMPLATE_ANSWER_ONLY = (
    "Problem: {problem}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n"
    "Correct final answer: {answer}\n"
    "=== Reference Solution End ==="
    f"{TEACHER_TRANSITION_PROMPT}\n"
    f"{_BOXED_INSTRUCTION}"
)

STUDENT_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    f"{_BOXED_INSTRUCTION}"
)

STUDENT_SYSTEM_MESSAGE = ""
TEACHER_SYSTEM_MESSAGE = ""
