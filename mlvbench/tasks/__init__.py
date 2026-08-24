"""Tasks.

Tasks allow reusing the trainer for different probing experiments by specifying

- How to obtain groung truth for a batch.
- The loss function.
- How to evaluate a probe.
"""

from mlvbench.tasks._base import Evaluator, Task
from mlvbench.tasks.binary_segmentation import BinarySegmentationTask

__all__ = [
    "BinarySegmentationTask",
    "Evaluator",
    "Task",
    "build_task",
]


def build_task(name: str) -> Task:
    """Build the task with the given name.

    Args:
        name: The name of the task.

    Returns:
        The task object.
    """
    match name:
        case "binary_segmentation":
            return BinarySegmentationTask()
        case _:
            raise ValueError(f"Unknown task: {name}")
