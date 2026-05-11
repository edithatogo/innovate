#!/bin/bash
#SBATCH --job-name=innovate-spack-smoke
#SBATCH --output=spack-smoke-%j.out
#SBATCH --error=spack-smoke-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -eu

: "${INNOVATE_SDIST:?set INNOVATE_SDIST to the source tarball path}"
: "${SPACK_PKG_PATH:?set SPACK_PKG_PATH to the candidate Spack package file}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

cp "$SPACK_PKG_PATH" "$workdir/py-innovate.py"
cp "$INNOVATE_SDIST" "$workdir/"

echo "scheduler=slurm"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "hostname=$(hostname)"
echo "spack_pkg=$(basename "$SPACK_PKG_PATH")"
echo "sdist=$(basename "$INNOVATE_SDIST")"

cat <<'EOF'
Expected follow-up:
- exercise the recipe with a real Spack environment
- capture `spack spec`, install logs, and a post-install smoke command
- archive the batch output in docs/source/_static/hpc_packaging/evidence/
EOF
