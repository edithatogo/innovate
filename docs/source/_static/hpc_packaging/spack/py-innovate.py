from spack.package import PythonPackage


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
