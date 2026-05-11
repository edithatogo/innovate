#!/bin/bash
#SBATCH --job-name=innovate-easybuild-smoke
#SBATCH --output=easybuild-smoke-%j.out
#SBATCH --error=easybuild-smoke-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -eu

: "${INNOVATE_SDIST:?set INNOVATE_SDIST to the source tarball path}"
: "${EASYCONFIG_PATH:?set EASYCONFIG_PATH to the candidate EasyBuild file}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

cp "$EASYCONFIG_PATH" "$workdir/innovate-0.5.0.eb"
cp "$INNOVATE_SDIST" "$workdir/"

echo "scheduler=slurm"
echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "hostname=$(hostname)"
echo "easyconfig=$(basename "$EASYCONFIG_PATH")"
echo "sdist=$(basename "$INNOVATE_SDIST")"

cat <<'EOF'
Expected follow-up:
- run the easyconfig in a real EasyBuild module environment
- capture module load output and sanity-check output
- archive the batch output in docs/source/_static/hpc_packaging/evidence/
EOF
