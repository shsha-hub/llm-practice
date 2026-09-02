"""학습과 추론에서 공유하는 생활 도우미 로직."""

from .config import MODEL_ID, ROOT, TASKS, generation_prompt, training_pair, user_text

__all__ = [
    "MODEL_ID",
    "ROOT",
    "TASKS",
    "generation_prompt",
    "training_pair",
    "user_text",
]

