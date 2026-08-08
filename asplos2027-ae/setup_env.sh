#!/usr/bin/env bash
# TritonParse — ASPLOS'27 Artifact Evaluation
# Build the environment, installing conda itself first if the host does not have one.
#
# environment.yml already reduces installation to a single command, but it still assumes
# conda is present.  This script removes that assumption: on a machine with nothing but a
# shell and an NVIDIA driver it fetches a conda distribution, installs it privately, and
# builds the environment from the manifest.
#
#     ./setup_env.sh
#
# It is safe to re-run.  An existing conda is reused rather than replaced, and an existing
# tritonparse-ae environment is left alone.
#
# WHAT IT WILL AND WILL NOT TOUCH.  Nothing is installed system-wide and no shell startup
# file is modified: we deliberately do not run `conda init`, because a reviewer's ~/.bashrc
# is not ours to edit.  A conda that this script installs goes under the artifact
# directory and is removed by deleting it.  If conda is already on PATH, that one is used
# as-is.
#
# Environment variables:
#   AE_CONDA_HOME    where to install conda if we have to        (default: <artifact>/.conda)
#   AE_CONDA_DISTRO  miniforge (default) or miniconda            -- see below
#   AE_ENV_NAME      name of the environment to create           (default: tritonparse-ae)
# It writes the resolved environment prefix to .env-prefix, which run_all.sh reads.
#
#   AE_KEEP_EXISTING_ENV=1
#                    reuse an existing environment instead of rebuilding it
#   AE_FORCE_INSTALL_CONDA=1
#                    install our own conda even if the host already has a usable one
#   AE_ACCEPT_ANACONDA_TOS=1
#                    accept Anaconda's Terms of Service non-interactively; only ever
#                    needed with AE_CONDA_DISTRO=miniconda
#
# WHY MINIFORGE IS THE DEFAULT.  Miniforge and Miniconda both give you the same `conda`
# command.  They differ in which channels they are configured for, and that difference is
# not cosmetic here: Miniconda ships with Anaconda's `defaults` channels enabled, and a
# recent conda refuses to solve anything until their commercial Terms of Service have been
# accepted --
#
#     CondaToSNonInteractiveError: Terms of Service have not been accepted ...
#
# even though environment.yml asks only for conda-forge.  Miniforge is configured for
# conda-forge already, so the manifest resolves with no terms to accept and no channel
# configuration to repair.  Accepting a licence is not something a setup script should do
# quietly on a reviewer's behalf, so the default avoids needing to.  Miniconda remains
# available with AE_CONDA_DISTRO=miniconda for anyone who wants it.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${AE_CONDA_HOME:-$HERE/.conda}"
DISTRO="${AE_CONDA_DISTRO:-miniforge}"
ENV_NAME="${AE_ENV_NAME:-tritonparse-ae}"
MANIFEST="$HERE/environment.yml"

# Pinned rather than "latest": a reviewer installing this in September should get the
# distribution we tested in August, and a checksum is only meaningful against a fixed
# file.  Both were downloaded and verified on 2026-08-06.
# A conda older than this cannot solve current conda-forge repodata.  Machines often
# carry an ancient /usr/bin/conda from a distribution package; reusing it produces a
# failure that looks like a network or manifest problem rather than a stale tool.
MIN_CONDA_MAJOR=23

MINIFORGE_VER="26.3.2-3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VER}/Miniforge3-Linux-x86_64.sh"
MINIFORGE_SHA="848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c"

MINICONDA_VER="py314_26.5.3-2"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VER}-Linux-x86_64.sh"
MINICONDA_SHA="80bc27f13c4de90f10e387aa45e864de4f0860692c1221aef5900009a2b55302"

step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m✗\033[0m %s\n' "$1" >&2; }
die()  { bad "$1"; exit 1; }

cat <<BANNER
================================================================================
 TritonParse — ASPLOS'27 Artifact Evaluation :: environment setup
================================================================================
BANNER

# --------------------------------------------------------------------------- 1. conda
step "1/3" "Locate conda"

[[ -f "$MANIFEST" ]] || die "environment.yml not found next to this script ($MANIFEST)"

CONDA=""
if [[ -x "$CONDA_HOME/bin/conda" ]]; then
    CONDA="$CONDA_HOME/bin/conda"                 # one we installed on an earlier run
    ok "reusing the conda this script installed: $CONDA"
elif [[ "${AE_FORCE_INSTALL_CONDA:-0}" == "1" ]]; then
    echo "  AE_FORCE_INSTALL_CONDA=1 — ignoring any conda already on PATH."
else
    # `conda` is often a shell function that dispatches to $CONDA_EXE; resolve to the
    # real binary so this works the same under `bash script.sh` as interactively.
    found="${CONDA_EXE:-$(command -v conda 2>/dev/null || true)}"
    if [[ -n "$found" && -x "$found" ]]; then
        ver="$("$found" --version 2>/dev/null | awk '{print $2}')"
        major="${ver%%.*}"
        if [[ "${major:-0}" =~ ^[0-9]+$ ]] && (( major >= MIN_CONDA_MAJOR )); then
            CONDA="$found"
            ok "using the conda already on PATH: $CONDA (conda $ver)"
        else
            echo "  Found conda $ver at $found, which is too old to solve current"
            echo "  conda-forge repodata (need $MIN_CONDA_MAJOR or newer). Leaving it alone."
        fi
    fi
fi

if [[ -z "$CONDA" ]]; then
    echo "  Installing $DISTRO privately under:"
    echo "      $CONDA_HOME"

    arch="$(uname -m)"
    [[ "$(uname -s)" == "Linux" && "$arch" == "x86_64" ]] || die \
        "this script pins a Linux x86-64 installer and this host is $(uname -s)/$arch;
   install conda yourself and re-run, or use the virtualenv path in README.md"

    case "$DISTRO" in
        miniforge) url="$MINIFORGE_URL"; sha="$MINIFORGE_SHA" ;;
        miniconda) url="$MINICONDA_URL"; sha="$MINICONDA_SHA" ;;
        *) die "AE_CONDA_DISTRO must be 'miniforge' or 'miniconda', not '$DISTRO'" ;;
    esac

    installer="$(mktemp -t conda-installer-XXXXXX.sh)"
    trap 'rm -f "$installer"' EXIT

    echo "  Downloading $(basename "$url") ..."
    command -v curl >/dev/null 2>&1 || die "curl is required to download the installer"
    curl -fsSL -o "$installer" "$url" || die "download failed: $url"

    # A truncated or substituted installer would otherwise fail later, in a way that
    # looks like a problem with the artifact rather than with the download.
    if command -v sha256sum >/dev/null 2>&1; then
        got="$(sha256sum "$installer" | cut -d' ' -f1)"
        [[ "$got" == "$sha" ]] || die "checksum mismatch for $(basename "$url")
   expected $sha
   got      $got"
        ok "checksum verified"
    else
        echo "  note: sha256sum not available, skipping checksum verification"
    fi

    bash "$installer" -b -p "$CONDA_HOME" >/tmp/tp-ae-conda-install.log 2>&1 \
        || die "installer failed — see /tmp/tp-ae-conda-install.log"
    CONDA="$CONDA_HOME/bin/conda"
    ok "installed $DISTRO $("$CONDA" --version 2>/dev/null) into $CONDA_HOME"
fi

# ------------------------------------------------------------------------ 2. the env
step "2/3" "Build the '$ENV_NAME' environment from environment.yml"

EXISTS=0
"$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME" && EXISTS=1

if [[ "$EXISTS" == "1" && "${AE_KEEP_EXISTING_ENV:-0}" == "1" ]]; then
    ok "'$ENV_NAME' already exists — reusing it (AE_KEEP_EXISTING_ENV=1)"
    echo "  Note that it is not checked against environment.yml; if the manifest has"
    echo "  changed since, re-run without AE_KEEP_EXISTING_ENV to rebuild."
else
    if [[ "$EXISTS" == "1" ]]; then
        # Rebuilding by default is the safer error: a half-built or drifted environment
        # produces failures in the checks that look like the artifact's fault and cost
        # far more to diagnose than the few minutes a rebuild takes.
        printf '  \033[33m⚠ WARNING\033[0m  an environment named %s already exists and will be\n' "$ENV_NAME"
        echo "             REPLACED. Anything you installed into it by hand is lost."
        echo "             Keep it instead with:  AE_KEEP_EXISTING_ENV=1 $0"
        if [[ -t 0 ]]; then
            echo "             Continuing in 5 seconds — press Ctrl-C to stop."
            sleep 5
        fi
        "$CONDA" env remove -n "$ENV_NAME" -y >/dev/null 2>&1 \
            || "$CONDA" env remove -n "$ENV_NAME" >/dev/null 2>&1 || true
        ok "removed the previous '$ENV_NAME'"
    fi
    echo "  This downloads roughly 5 GB and takes a few minutes."
    create_log="$(mktemp -t tp-ae-envcreate-XXXXXX.log)"

    # `-y` is not accepted by every conda vintage.  Ask --help rather than discovering it
    # by way of a failed five-gigabyte download.
    YES=()
    if "$CONDA" env create --help 2>&1 | grep -qE '(^|[[:space:],])-y([[:space:],]|$)'; then
        YES=(-y)
    fi
    create() {
        "$CONDA" env create -f "$MANIFEST" -n "$ENV_NAME" ${YES[@]+"${YES[@]}"} \
            >"$create_log" 2>&1
    }

    if ! create; then
        # The one failure worth handling rather than just reporting, because it is about
        # the reviewer's conda configuration and not about this artifact.
        if grep -q "CondaToSNonInteractive" "$create_log"; then
            if [[ "${AE_ACCEPT_ANACONDA_TOS:-0}" == "1" ]]; then
                echo "  Accepting Anaconda's Terms of Service (AE_ACCEPT_ANACONDA_TOS=1) ..."
                for ch in main r; do
                    "$CONDA" tos accept --override-channels \
                        --channel "https://repo.anaconda.com/pkgs/$ch" >/dev/null 2>&1 || true
                done
                create || { tail -20 "$create_log" >&2; die "environment creation failed — full log: $create_log"; }
            else
                bad "conda will not solve until Anaconda's Terms of Service are accepted"
                cat <<'TOS' >&2

  This is conda's own gate, not the artifact's: `defaults` is conda's built-in
  channel list, so it applies even though environment.yml asks only for
  conda-forge.  Three ways forward, in order of least commitment:

    1. Use Miniforge instead, which is configured for conda-forge and so
       never reaches those channels at all:

           AE_FORCE_INSTALL_CONDA=1 AE_CONDA_HOME=./.conda ./setup_env.sh

       That installs one privately, next to this script, whatever conda the
       host already has.  Deleting ./.conda undoes it.

    2. Accept the two channels once, yourself or via this script:

           conda tos accept --override-channels \
               --channel https://repo.anaconda.com/pkgs/main
           conda tos accept --override-channels \
               --channel https://repo.anaconda.com/pkgs/r

           AE_ACCEPT_ANACONDA_TOS=1 ./setup_env.sh

    3. Skip conda entirely and use the virtualenv path in README.md.  It needs
       Python >= 3.11 on the host and gives you everything except Node, which
       only the optional interface build wants.

TOS
                exit 1
            fi
        else
            tail -20 "$create_log" >&2
            die "environment creation failed — full log: $create_log"
        fi
    fi
    ok "'$ENV_NAME' created"
    rm -f "$create_log"
fi

# --------------------------------------------------------------------------- 3. next
step "3/3" "Activate it and run the checks"

ENV_PREFIX="$("$CONDA" env list | awk -v n="$ENV_NAME" '$1==n {print $NF}' | head -1)"

# run_all.sh has to put this environment on PATH without a shell hook, and a human
# re-reading it later should not have to re-derive it either.  One line, one path.
if [[ -n "$ENV_PREFIX" ]]; then
    printf '%s\n' "$ENV_PREFIX" > "$HERE/.env-prefix"
    printf '%s\n' "$(dirname "$(dirname "$CONDA")")" > "$HERE/.conda-root"
fi

cat <<NEXT
  Activate the environment:

NEXT
if [[ "$CONDA" == "$CONDA_HOME/bin/conda" ]]; then
    # This conda was never `conda init`-ed, so its shell hook has to be sourced first.
    cat <<NEXT
      source $CONDA_HOME/etc/profile.d/conda.sh
      conda activate $ENV_NAME
NEXT
else
    cat <<NEXT
      conda activate $ENV_NAME
NEXT
fi

cat <<NEXT

  Then, from the repository root:

      ./asplos2027-ae/00_kick_the_tires.sh     # basic test, up to ~2 min
      ./asplos2027-ae/01_gpu_core.sh           # main check, up to ~8 min
      ./asplos2027-ae/02_review_ui.sh          # optional interface build

  If you prefer not to activate anything, the environment's interpreter works
  directly:

      ${ENV_PREFIX:-<env prefix>}/bin/python

================================================================================
NEXT
