#!/usr/bin/env bash
# TritonParse — ASPLOS'27 Artifact Evaluation
# C3, second half: build the Review interface locally and open it on the shipped trace.
#
# The other two scripts validate the Review stage's *data layer* from the command line
# (`tritonparseoss diff` produces the per-IR-stage comparison and the per-Python-line
# attribution).  What they cannot check is the thing paper 3.3 actually describes: an
# interactive interface for inspecting a kernel across IR levels.  That is a visual
# claim, so this script does the part a machine can do -- build the interface from source
# and prove the build is usable -- and then hands you the file to look at.
#
# Optional.  Needs Node.js and npm; nothing else in the artifact does.  environment.yml
# installs them, so under the conda environment there is nothing extra to do.  If you
# built a plain virtualenv instead, or would rather not build anything at all, the same
# interface is hosted at https://meta-pytorch.org/tritonparse/ and takes the same file.
#
# Usage:  ./02_review_ui.sh
#
# Expected wall time: up to about 1 minute once npm has a cache (measured: 8 s of
# npm ci, 3.6 s of build; roughly double on a slower machine), plus the download of
# roughly 300 MB of npm packages the first time.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture this run to a file even when it is launched on its own, and identify the tree
# it ran from.  Both have to happen before anything prints.
# shellcheck source=scripts/ae_logging.sh
source "$HERE/scripts/ae_logging.sh"
ae_begin_log "review_ui" "$@"
# shellcheck source=scripts/ae_provenance.sh
source "$HERE/scripts/ae_provenance.sh"
AE_TREE="$(ae_provenance "$HERE")"; export AE_TREE


# Use the artifact's environment whether or not the caller activated it.
# shellcheck source=scripts/activate_env.sh
source "$HERE/scripts/activate_env.sh"
REPO_ROOT="${TRITONPARSE_REPO:-$(cd "$HERE/.." && pwd)}"
WEBSITE="$REPO_ROOT/website"
TRACE="$(find "$HERE/traces/reference/parsed" -name '*_mapped.ndjson.gz' 2>/dev/null | head -1)"

step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }
ok()   { printf '  \033[32m✓ PASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31m✗ FAIL\033[0m  %s\n' "$1"; }

cat <<BANNER
================================================================================
 TritonParse — ASPLOS'27 Artifact Evaluation :: Review interface (C3, optional)
================================================================================
BANNER

# The environment this ran in, in the console and therefore in the log.  Most reports of
# "this gave me a different number" are answered by these lines alone.
if command -v python >/dev/null 2>&1; then
    python "$HERE/scripts/ae_env.py" ${AE_LOG_DIR:+--full "$AE_LOG_DIR"} 2>/dev/null || true
fi

step "1/3" "Prerequisites"

if [[ ! -d "$WEBSITE" ]]; then
    bad "no website/ directory under $REPO_ROOT"
    echo "  Set TRITONPARSE_REPO to the repository root and re-run."
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    bad "npm not found"
    cat <<'NONODE'

  This is the only part of the artifact that needs Node.js, and it is optional.
  environment.yml installs it, so the quickest fix is to use that environment:

      conda env create -f environment.yml && conda activate tritonparse-ae

  Otherwise install Node yourself (any recent LTS; validated on v24), or skip the
  local build and use the hosted instance:

      https://meta-pytorch.org/tritonparse/

  It is the same interface, and it takes the same trace file. All processing
  happens in your browser; nothing is uploaded.
NONODE
    exit 1
fi
ok "node $(node -v 2>/dev/null), npm $(npm -v 2>/dev/null)"

# ------------------------------------------------------------------------- build
step "2/3" "Build the interface from source"

echo "  (first run downloads ~300 MB of npm packages)"
if ! (cd "$WEBSITE" && npm ci --no-audit --no-fund) >/tmp/tp-ui-install.log 2>&1; then
    bad "npm ci failed — see /tmp/tp-ui-install.log"
    tail -15 /tmp/tp-ui-install.log | sed 's/^/    /'
    exit 1
fi
ok "dependencies installed"

# build:single inlines the whole application into one HTML file, so what you open has
# no local server and no external assets behind it.
if ! (cd "$WEBSITE" && npm run build:single) >/tmp/tp-ui-build.log 2>&1; then
    bad "build failed — see /tmp/tp-ui-build.log"
    tail -20 /tmp/tp-ui-build.log | sed 's/^/    /'
    exit 1
fi

HTML="$WEBSITE/dist/standalone.html"
if [[ ! -f "$HTML" ]]; then
    bad "build reported success but $HTML is missing"
    exit 1
fi
SIZE=$(stat -c%s "$HTML" 2>/dev/null || stat -f%z "$HTML")
if [[ "$SIZE" -lt 200000 ]]; then
    bad "standalone.html is only $SIZE bytes — the application does not look inlined"
    exit 1
fi
ok "standalone.html built ($(numfmt --to=iec "$SIZE" 2>/dev/null || echo "$SIZE B"))"

# Self-containment is the property that makes this openable straight from disk.
EXTERNAL=$(grep -oE '(src|href)="(\./|/|assets/)[^"]*"' "$HTML" 2>/dev/null | wc -l)
if [[ "$EXTERNAL" -gt 0 ]]; then
    echo "  note: $EXTERNAL relative asset reference(s) remain; serve dist/ over HTTP if"
    echo "        opening the file directly does not work."
else
    ok "self-contained — no local assets to serve alongside it"
fi

# ------------------------------------------------------------------------- inspect
step "3/3" "What to look at"

cat <<INSPECT
  Open this file in a browser:

      file://$HTML

  Then load the trace that ships with the artifact:

      ${TRACE:-<trace not found — run 00_kick_the_tires.sh first>}

  Paper 3.3 claims the Review stage lets a developer inspect one kernel across every
  IR level at once.  Things to check, and where they appear in the paper:

    * the IR panes -- Python source, TTIR, TTGIR, LLIR and PTX for the same kernel,
      side by side  (3.3, and the pipeline figure in 2)
    * click a line in one pane and see the corresponding lines highlight in the
      others -- this is the source mapping the Reconstruct stage builds  (3.2, 3.3)
    * the launch view -- the five compilations in this trace, with their differing
      constexpr values and launch counts  (3.3)
    * the diff view -- select two compilations of matmul_kernel and compare them; the
      same comparison 00_kick_the_tires.sh prints on the console is what drives the
      two case studies in 5.1 and 5.2

  Nothing is uploaded: the file is a static page and the trace is read locally.

  The hosted build of the same interface is at https://meta-pytorch.org/tritonparse/
  if you want to cross-check that a local build behaves identically.
================================================================================
INSPECT
