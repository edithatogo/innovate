.. innovate documentation master file, created by
   sphinx-quickstart on Mon Jul  7 15:24:15 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Innovate: Innovation Diffusion Modeling Library
===============================================

Welcome to **innovate**, a comprehensive Python library for modeling innovation diffusion processes. Whether you're analyzing technology adoption, market penetration, or social innovation spread, innovate provides the tools you need.

.. note::
   **Quick Start**: Jump to our :doc:`tutorials_comprehensive` to get started with hands-on examples, or explore the complete :doc:`api_reference` for detailed technical documentation.

🚀 **Key Features**

* **Comprehensive Model Suite**: Bass, Gompertz, Logistic, and multi-product competition models
* **Advanced Analytics**: Agent-based modeling, policy intervention analysis, and hype cycle modeling  
* **Flexible Backends**: NumPy and JAX support for different performance needs
* **Rich Visualizations**: Built-in plotting tools for diffusion curves and competitive dynamics
* **Real-world Examples**: Extensive Jupyter notebook tutorials with case studies

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   tutorials_comprehensive
   api_reference

.. toctree::
   :maxdepth: 2
   :caption: Core Documentation

   innovate
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   innovate.diffuse.rst
   innovate.compete.rst
   innovate.substitute.rst
   innovate.hype.rst
   innovate.fail.rst
   innovate.adopt.rst
   innovate.path_dependence.rst
   innovate.policy.rst
   innovate.abm.rst

.. toctree::
   :maxdepth: 2
   :caption: Utilities & Extensions

   innovate.plots.rst
   innovate.utils.rst
   innovate.fitters.rst
   innovate.fitters.diagnostics_contract.rst
   innovate.arrow_interchange.rst
   innovate.rust_bindings.rst
   innovate.plugins.rst
   innovate.stability.rst
   innovate.kernel.rst
   innovate.base.rst

.. toctree::
   :maxdepth: 1
   :caption: Architecture & Roadmap

   ../architecture_principles
   ../architecture_modernization_roadmap
   ../adr/index

.. toctree::
   :maxdepth: 1
   :caption: Development & Contributing

   ../testing_strategy

📖 **Quick Examples**

Basic Bass Model
~~~~~~~~~~~~~~~

.. code-block:: python

   from innovate import BassModel, ScipyFitter
   
   # Fit Bass model to adoption data
   model = BassModel()
   fitter = ScipyFitter()
   fitter.fit(model, t=[1,2,3,4,5], y=[10,25,50,85,120])
   
   # Predict future adoption
   predictions = model.predict([6,7,8,9,10])

Competitive Dynamics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from innovate.compete import MultiProductDiffusionModel
   
   # Model competition between two products
   model = MultiProductDiffusionModel(
       p=[0.02, 0.03],  # Innovation coefficients
       Q=[[0.1, 0.05], [0.03, 0.1]],  # Competition matrix  
       m=[1000, 800]    # Market potentials
   )
   
   adoption = model.predict(t=[1,2,3,4,5])

Policy Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   from innovate.policy import PolicyIntervention
   
   # Analyze subsidy impact on adoption
   policy = PolicyIntervention(bass_model)
   policy.add_subsidy(start_time=2, end_time=5, effect_size=0.3)
   
   # Compare scenarios
   baseline = model.predict(time_points)
   with_policy = policy.apply_interventions(time_points)

Canonical imports
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter
   from innovate.compete import MultiProductDiffusionModel, LotkaVolterraModel
   from innovate.substitute import FisherPryModel, NortonBassModel
   from innovate.backends import use_backend

🎯 **Use Cases**

* **Technology Adoption**: Model smartphone, EV, or renewable energy adoption
* **Market Analysis**: Analyze competitive dynamics and market penetration
* **Policy Impact**: Assess effects of subsidies, regulations, and incentives
* **Innovation Strategy**: Optimize product launch timing and market entry
* **Academic Research**: Study diffusion processes in social sciences and economics

📊 **Model Types**

=================== ================================ ========================
Model               Best For                         Key Strengths
=================== ================================ ========================
Bass Model          Standard innovation adoption     Word-of-mouth dynamics
Gompertz Model      Slow initial growth patterns     Infrastructure adoption
Logistic Model      Resource-constrained growth      Population dynamics
Multi-Product       Competitive market scenarios     Market competition
Agent-Based         Complex heterogeneous systems    Individual behaviors
=================== ================================ ========================

💡 **Getting Help**

* **Documentation**: Complete API reference and tutorials
* **Examples**: 15+ Jupyter notebooks with real-world case studies
* **GitHub**: Source code, issues, and contributions at `github.com/doughnutsz/innovate <https://github.com/doughnutsz/innovate>`_
* **Community**: Discussion and support through GitHub Issues

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
