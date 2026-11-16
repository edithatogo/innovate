# Reproducing the isolated JOSS draft environment

This recipe provisions owned directories on Linux x86_64 and invokes the
unchanged [draft wrapper](../../../joss/render.py). It does not install into
system TeX, qualify other platforms, submit a paper or supply author approval.
The inspected official [JOSS local-build guidance](https://joss.readthedocs.io/en/latest/paper.html#locally)
permits Inara's Makefile after installing its dependencies. The pinned
[Inara Dockerfile](https://github.com/openjournals/inara/blob/88ee5d656f147d801448ce77ad8375619d383df6/Dockerfile)
provides the Pandoc version, TeX Live archive and additional package requirements.

The recipe uses the official TeX Live 2024 base and recommended LaTeX collections,
LuaLaTeX and the pinned template's additional packages. The
[package builder](build_pdf_environment.py) resolves the archived database and
requires a fresh setup root, pins the database's SHA-256 before any writes, and
checks every downloaded container's size and SHA-512 before safe extraction.
The [configuration helper](configure_pdf_environment.py) verifies the Inara/Hack
revisions and Pandoc version, checks every copied font/license against its Git
blob and rejects extra files, then copies those fonts into its own tree. It
requires the TeX executables to resolve inside that owned tree and builds its
own LuaLaTeX format and font cache. Each helper refuses an existing installation
or configuration directory. Neither helper changes the manuscript or renderer.

Prerequisites are Git, curl, GNU Make, Perl and compatible Linux x86_64 system
libraries. The original PDF environment's package bootstrap used Python 3.12.14;
configuration and the repository renderer used Python 3.15.2. The later guarded
setup replay used Python 3.15.2 throughout. The helpers require Python's safe
tar extraction filter, available in Python 3.12 or newer. The TeX archive and
Pandoc binary are Linux x86_64 builds; this is not native Windows qualification.

## Obtain the pinned inputs

Use a fresh child directory of an existing owned parent, with no spaces or shell
metacharacters. Set `joss_repo` to
the reviewed checkout and `joss_python` to its Python 3.15.2 executable. These
commands describe fresh acquisition; this execution reused the already acquired
and verified Pandoc/Hack/Inara sources, as recorded in the environment receipt.

```sh
set -eu
joss_repo=/path/to/reviewed/innovate
joss_tools=/path/to/new/innovate-joss-tools
joss_python=/path/to/python3.15
joss_bootstrap_python=python3.12
joss_packet="$joss_repo/conductor/tracks/research_software_registry_readiness_20260721/evidence/evidence-acquisition-20261116/render"
if [ -e "$joss_tools" ] || [ -L "$joss_tools" ]; then
  printf '%s\n' "Refusing existing tools directory: $joss_tools" >&2
  exit 1
fi
mkdir "$joss_tools"

curl --fail --location --output "$joss_tools/texlive-2024-final.tlpdb.xz" \
  https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2024/tlnet-final/tlpkg/texlive.tlpdb.xz
curl --fail --location --output "$joss_tools/pandoc-3.6.4-linux-amd64.tar.gz" \
  https://github.com/jgm/pandoc/releases/download/3.6.4/pandoc-3.6.4-linux-amd64.tar.gz

"$joss_bootstrap_python" - "$joss_tools" <<'PY'
import hashlib
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {
    "texlive-2024-final.tlpdb.xz": "60b9f8f8893d591d220d2f4f8dda3f686caa165efe78aa0ccc8dec80d48799aa",
    "pandoc-3.6.4-linux-amd64.tar.gz": "5def6e1ff535e397becce292ee97767a947306150b9fb1488003b67ac3417c5e",
}
for name, sha256 in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha256:
        raise SystemExit(f"Checksum mismatch: {name}")
PY

tar -xzf "$joss_tools/pandoc-3.6.4-linux-amd64.tar.gz" -C "$joss_tools"
git init "$joss_tools/inara"
git -C "$joss_tools/inara" fetch --depth 1 https://github.com/openjournals/inara.git 88ee5d656f147d801448ce77ad8375619d383df6
git -C "$joss_tools/inara" checkout --detach FETCH_HEAD
git init "$joss_tools/hack"
git -C "$joss_tools/hack" fetch --depth 1 https://github.com/source-foundry/Hack.git b47d21f808fbf4784565feccde2213ef2af924b3
git -C "$joss_tools/hack" checkout --detach FETCH_HEAD

"$joss_bootstrap_python" "$joss_packet/build_pdf_environment.py" \
  --root "$joss_tools/tex" --database "$joss_tools/texlive-2024-final.tlpdb.xz"
"$joss_python" "$joss_packet/configure_pdf_environment.py" \
  --output "$joss_tools/configured" --texlive "$joss_tools/tex/texlive" \
  --inara "$joss_tools/inara" --hack "$joss_tools/hack" \
  --pandoc "$joss_tools/pandoc-3.6.4/bin/pandoc"
```

## Execute the unchanged draft wrapper

The helper records the exact process-local environment in `render-env.json`.
Load it only for the render subprocess. `TEXMFCACHE=.` creates the font cache in
the owned Inara checkout; the wrapper permits this cache while rejecting
changes to upstream tracked files. Use a fresh output directory for each attempt.

```sh
"$joss_python" - "$joss_repo" "$joss_tools" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
repo, tools = map(Path, sys.argv[1:])
env = os.environ.copy()
env.update(json.loads((tools / "configured/render-env.json").read_text()))
command = [
    sys.executable,
    str(repo / "conductor/tracks/research_software_registry_readiness_20260721/joss/render.py"),
    "--inara", str(tools / "inara"),
    "--pandoc", str(tools / "pandoc-3.6.4/bin/pandoc"),
    "--output-dir", str(tools / "draft-output"),
    "--source-date-epoch", "1794787200",
]
raise SystemExit(subprocess.run(command, cwd=repo, env=env).returncode)
PY
```

The wrapper first checks the source/edition identities and always forces draft
mode. Its output receipt separates bibliography parsing, official TeX, PDF
compilation and the still-pending visual check. Render every PDF page with
`pdftoppm -r 150 -png`, inspect it, and record that review against the PDF hash in
a separate receipt. Do not rewrite the renderer's original receipt to invent a
visual result. Source changes require claim reassessment before another render.

This is the local Makefile route, not a build or execution of the official
container. ConTeXt/JATS, PDF/A conformance and tagged-PDF accessibility are not
qualified by these commands. Matching the recorded software revisions does not
promise identical PDF bytes on a different host; retain the actual environment
and output identities for each run.
