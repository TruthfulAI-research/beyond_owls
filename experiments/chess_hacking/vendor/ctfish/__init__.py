"""ctfish: chess agent specification gaming (arXiv:2502.13295)."""

from .agent import CommandRecord, CtfishState, ctfish_agent
from .config import (
    GENERATE_CONFIG_PRESETS,
    QWEN3_THINKING_PARAMS,
    generate_config,
    resolve_generate_config,
)
from .ctfish import ctfish
from .prompts import TASK_PROMPTS
from .scorer import (
    engine_assisted_moves,
    escalation_judge,
    moves_submitted,
    stage_judge,
    truncated_generations,
    won_game,
)

__all__ = [
    "CommandRecord",
    "CtfishState",
    "GENERATE_CONFIG_PRESETS",
    "QWEN3_THINKING_PARAMS",
    "TASK_PROMPTS",
    "ctfish",
    "ctfish_agent",
    "engine_assisted_moves",
    "escalation_judge",
    "generate_config",
    "moves_submitted",
    "resolve_generate_config",
    "stage_judge",
    "truncated_generations",
    "won_game",
]
