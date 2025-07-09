
innovate.abm
============

The ``innovate.abm`` module provides a framework for creating Agent-Based Models (ABMs) of innovation diffusion. It is built on top of the `mesa` library.

.. automodule:: innovate.abm
   :members:

Pre-configured Scenarios
------------------------

The module includes three pre-configured ABM scenarios that can be used to model different aspects of innovation diffusion.

Competitive Diffusion
~~~~~~~~~~~~~~~~~~~~~

The ``CompetitiveDiffusionModel`` simulates a scenario where multiple innovations are competing for adoption in a population of agents. The agents' adoption decisions are based on the choices of their neighbors.

Sentiment-Driven Hype Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SentimentHypeModel`` models the impact of sentiment on the adoption of an innovation. Agents' decisions are influenced by both the adoption status and the sentiment of their neighbors.

Disruptive Innovation
~~~~~~~~~~~~~~~~~~~~~

The ``DisruptiveInnovationModel`` simulates the competition between an established incumbent product and a new disruptive one. The disruptive innovation starts with lower performance but improves over time.

Example Usage
-------------

The following example demonstrates how to use the pre-configured ABM scenarios.

.. literalinclude:: ../../examples/abm_examples.py
   :language: python

