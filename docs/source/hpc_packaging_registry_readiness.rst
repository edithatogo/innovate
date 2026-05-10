HPC packaging and registry readiness
====================================

Purpose
-------

This dossier defines the package surfaces, dependency variants, installation
checks, and registry evidence needed before ``innovate`` can be proposed for
HPC-oriented distribution channels such as Spack, EasyBuild, HPSF, or E4S.

The current state is readiness planning, not a registry claim. Spack,
EasyBuild, HPSF, and E4S publication still requires HPC-specific scheduler and
performance-portability evidence, but the repository now contains concrete
package sketches and install-smoke artifacts for the supported surfaces.

See also :doc:`hpc_registry_contract` for the submission contract and
:doc:`polyglot_registry_plan` for the broader registry sequencing plan.

Concrete artifacts
------------------

The repository now carries the package and evidence bundle directly under
``docs/source/_static/hpc_packaging/``:

* ``spack/py-innovate.py`` - candidate Spack recipe sketch;
* ``easybuild/innovate-0.5.0.eb`` - candidate EasyBuild easyconfig sketch;
* ``evidence/python-install.log`` - clean wheel install into an isolated
  Python environment;
* ``evidence/python-smoke.log`` - import and dependency-check smoke log;
* ``evidence/r-build.log`` - R package build evidence;
* ``evidence/r-check.log`` - R CMD check evidence;
* ``evidence/rust-test.log`` - Rust binding test evidence;
* ``evidence/uv-build.log`` - repository wheel/sdist build evidence;
* ``evidence/julia-installed-smoke.log`` - installed-package Julia bridge
  smoke evidence.

Install surfaces
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Surface
     - Required evidence
     - Packaging notes
   * - Python package surface
     - Build wheel and sdist, install under Python 3.10 through 3.14, run
       ``python -m pip check``, and import ``innovate``.
     - Core dependencies map to Spack Python package names such as
       ``py-numpy``, ``py-scipy``, ``py-pandas``, ``py-pyarrow``,
       ``py-statsmodels``, ``py-mesa``, ``py-networkx``, ``py-ndlib``,
       ``py-jitcdde``, ``py-sympy``, ``py-ruptures``,
       ``py-pymannkendall``, ``py-pytensor``, and
       ``py-typing-extensions``.
   * - Rust crate and native slices
     - Build and run ``cargo test --manifest-path bindings/rust/Cargo.toml``.
     - The ``+rust`` variant requires ``rust`` and ``cargo`` and should remain
       optional until promoted native slices are proven across supported
       compilers and architectures.
   * - Optional JAX/XLA extras
     - Install the JAX extra and run a guarded accelerator smoke test on CPU
       first, then on GPU where an accelerator runner exists.
     - The ``+jax`` variant should depend on ``py-jax`` and ``py-jaxlib`` and
       record the XLA backend, driver, and scheduler context.
   * - Language binding surfaces
     - Run guarded binding smoke tests for Julia, R, and TypeScript packages.
     - The ``+bindings`` variant should install only bridge-backed binding
       tests that can run without network access or registry publishing.

Deployment options
------------------

CPU-only deployment
   Installs the Python package with core scientific dependencies, no optional
   accelerator packages, and no Rust-native requirement. The smoke evidence is
   the wheel/sdist install log, ``python -m pip check``, and a minimal import
   or kernel call on a login-node-like environment.

GPU/XLA deployment
   Extends CPU-only deployment with ``py-jax`` and ``py-jaxlib``. The smoke
   evidence must identify the XLA backend, CUDA or ROCm stack when present,
   scheduler allocation metadata, and whether the call ran on CPU fallback or a
   real GPU device.

Mixed Rust/Python bridge deployment
   Enables Rust-native slices while preserving Python bridge fallback for
   operations not yet promoted. Evidence must show the Rust crate build,
   bridge-backed calls, and a failure mode where unavailable native capability
   discovery falls back without changing public API behavior.

Spack package candidate
-----------------------

The first Spack recipe should remain a candidate until it has been exercised in
CI and at least one Slurm or PBS environment. The recipe sketch below captures
the package shape and dependency variants without claiming upstream acceptance.

.. code-block:: python

   class PyInnovate(PythonPackage):
       """Spack candidate for innovate."""

       homepage = "https://github.com/edithatogo/innovate"
       pypi = "innovate/innovate-0.5.0.tar.gz"

       version("0.5.0", sha256="REPLACE_WITH_SDIST_SHA256")

       variant("+rust", default=False, description="Build Rust native slices")
       variant("+jax", default=False, description="Install optional JAX/XLA extras")
       variant("+bindings", default=False, description="Run binding smoke checks")
       variant("+docs", default=False, description="Install documentation dependencies")

       depends_on("python@3.10:", type=("build", "run"))
       depends_on("py-setuptools", type="build")
       depends_on("py-numpy@1.24:", type=("build", "run"))
       depends_on("py-scipy@1.10:", type=("build", "run"))
       depends_on("py-pandas@2:", type=("build", "run"))
       depends_on("py-pyarrow@14:", type=("build", "run"))
       depends_on("py-statsmodels@0.14:", type=("build", "run"))
       depends_on("py-mesa@2:", type=("build", "run"))
       depends_on("py-networkx@3:", type=("build", "run"))
       depends_on("py-ndlib@5.1:", type=("build", "run"))
       depends_on("py-jitcdde@1.8:", type=("build", "run"))
       depends_on("py-sympy@1.12:", type=("build", "run"))
       depends_on("py-ruptures@1.1:", type=("build", "run"))
       depends_on("py-pymannkendall@1.4:", type=("build", "run"))
       depends_on("py-pytensor@2.18:", type=("build", "run"))
       depends_on("py-typing-extensions@4.7:", type=("build", "run"))
       depends_on("rust@1.85:", when="+rust", type="build")
       depends_on("cargo", when="+rust", type="build")
       depends_on("py-jax@0.4.20:", when="+jax", type=("build", "run"))
       depends_on("py-jaxlib@0.4.20:", when="+jax", type=("build", "run"))

       def test(self):
           python("-m", "pip", "check")
           python("-c", "import innovate; print(innovate.__version__)")

EasyBuild easyconfig candidate
------------------------------

The first EasyBuild contribution should mirror the Spack evidence and keep
binding checks opt-in. The easyconfig candidate should start with the Python
wheel path and add module sanity checks before registry submission.

.. code-block:: python

   easyblock = 'PythonPackage'

   name = 'innovate'
   version = '0.5.0'

   toolchain = {'name': 'foss', 'version': '2024a'}
   download_dep_fail = True
   use_pip = True
   sanity_pip_check = True

   sanity_check_commands = [
       "python -m pip check",
       "python -c \"import innovate; print(innovate.__version__)\"",
   ]

   sanity_check_paths = {
       'files': [],
       'dirs': ['lib/python%(pyshortver)s/site-packages/innovate'],
   }

module sanity checks
   EasyBuild module sanity should verify imports, dependency resolution, and a
   guarded kernel call. Optional binding checks can be run after module load:
   ``julia --project=bindings/julia -e 'using Innovate'``, ``Rscript -e
   'library(jsonlite)'``, and ``npm test --prefix bindings/typescript``.

Install and smoke evidence
--------------------------

Every packaging candidate needs durable evidence stored with the release or CI
artifact. The repository now includes that evidence for the local build and
binding surfaces:

* CPU-only install evidence in ``evidence/python-install.log``;
* Python smoke evidence in ``evidence/python-smoke.log``;
* Rust evidence from ``evidence/rust-test.log``;
* Julia installed-package bridge evidence in
  ``evidence/julia-installed-smoke.log``;
* repository build evidence in ``evidence/uv-build.log``;
* JAX/XLA evidence with backend and device metadata when ``+jax`` is enabled;
* scheduler evidence from at least one Slurm or PBS batch run before an HPC
  registry claim is made;
* CPU, GPU, and mixed bridge smoke results before performance portability is
  claimed.

HPSF and E4S registry gates
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Target
     - Current status
     - Required next evidence
   * - HPSF candidacy
     - install and smoke evidence is now present; remaining evidence covers
       governance alignment, release durability, and HPC user support.
     - Add installer logs for scheduler-backed environments, support contacts,
       maintenance policy, and examples showing scheduler-aware deployment.
   * - E4S candidacy
     - package sketches and local evidence are present; remaining evidence
       covers performance portability and accelerator-aware validation.
     - Add accepted or reviewable Spack recipe evidence, CPU/GPU smoke logs,
       dependency variant results, and failure-mode diagnostics.

Result: no HPSF or E4S submission should be made until the package candidates
have passing install log and smoke log artifacts for CPU, GPU, and mixed
bridge deployment, plus a documented performance portability evidence bundle.

The current gate remains blocked until evidence exists.
Current smoke commands to preserve in the evidence bundle include
``python -c "import innovate; print(innovate.__version__)"`` for Python and
``Rscript -e`` for the R surface.
