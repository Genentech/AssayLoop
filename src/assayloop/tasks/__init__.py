"""The sequential-design task and the screen sets it runs on.

``ScreenRecord``, the loader that produces one, and ``gene_universe`` now live
in ``assaybench`` -- they describe the benchmark, so ``pip install assaybench``
is enough to score a policy the way the paper does. ``gene_batch`` holds the
``Task`` implementation, and ``screen_sets`` holds this repository's short
names for the benchmark's sets (``public``, ``public_train``, ...).

Everything the two moved modules used to export is re-exported here, so
``from assayloop.tasks import ScreenRecord, load_screens, gene_universe``
keeps working unchanged.
"""

from .gene_batch import (
    AssayBenchGeneBatchTask,
    ScreenRecord,
    make_task,
    screen_from_example,
)
from .screen_sets import (
    default_public_screen_set_path,
    default_public_validation_screen_set_path,
    gene_universe,
    load_screens,
)

__all__ = [
    "AssayBenchGeneBatchTask",
    "ScreenRecord",
    "screen_from_example",
    "make_task",
    "gene_universe",
    "load_screens",
    "default_public_screen_set_path",
    "default_public_validation_screen_set_path",
]
