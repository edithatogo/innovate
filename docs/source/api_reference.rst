API Reference
=============

This page is retained for compatibility. The canonical deep documentation now
lives in the Sphinx docs under ``docs/source``.

Stable public imports
---------------------

The recommended user-facing API is the package-level surface:

.. code-block:: python

   from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter
   from innovate.compete import MultiProductDiffusionModel, LotkaVolterraModel
   from innovate.substitute import CompositeDiffusionModel, FisherPryModel, NortonBassModel
   from innovate.ecosystem import ComplementaryGoodsModel
   from innovate.backends import use_backend

Read next
---------

- ``docs/source/index.rst`` for the canonical docs landing page
- ``docs/source/tutorials_comprehensive.rst`` for workflow-oriented guides
- ``docs/source/bindings.rst`` for language bindings
- ``docs/source/innovate.kernel.rst`` for the contract-first core API
- ``docs/source/innovate.arrow_interchange.rst`` for interchange details
