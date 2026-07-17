"""[deprecated] 请直接从 ``core.experiment`` 导入 ``ExperimentManager``。

.. code:: python

    from core.experiment import ExperimentManager, PlotSpec
"""

from core.experiment import ExperimentManager, PlotSpec  # noqa: F401 — 向后兼容

__all__ = ["ExperimentManager", "PlotSpec"]

