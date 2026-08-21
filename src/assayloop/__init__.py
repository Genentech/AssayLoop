"""assayloop — modular active-learning loops for screen prediction.

A task proposes a candidate pool, a model scores it, and an acquisition
function picks the next batch to measure. Those four abstractions and the
loop that drives them are not defined here: they live in the `assaybench`
package (`assaybench.core`), so that a third party can implement against
the benchmark without taking on this repo's model zoo, LLM clients and
paper scripts. They are re-exported below because this is where the
research code reaches for them.
"""

from assaybench.core import (
    AcquisitionFunction,
    HistoryEntry,
    Metric,
    Model,
    ModelPrediction,
    Observation,
    RunResult,
    SequentialLoop,
    StepRecord,
    Task,
)

__all__ = [
    "Task",
    "Model",
    "AcquisitionFunction",
    "Metric",
    "SequentialLoop",
    "Observation",
    "ModelPrediction",
    "StepRecord",
    "RunResult",
    "HistoryEntry",
]
