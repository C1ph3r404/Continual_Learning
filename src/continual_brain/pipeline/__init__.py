from continual_brain.pipeline.stage0_wernicke import train_wernicke_syntax_step
from continual_brain.pipeline.stage1_neocortex import train_neocortex_memory_step
from continual_brain.pipeline.stage2_reasoning import train_frontal_reasoning_step

__all__ = [
    "train_wernicke_syntax_step",
    "train_neocortex_memory_step",
    "train_frontal_reasoning_step"
]
