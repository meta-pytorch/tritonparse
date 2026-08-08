# TritonParse — ASPLOS'27 Artifact Evaluation
# Put the artifact's conda environment in front of whatever the caller happens to have.
#
# Sourced, not executed:
#
#     source "$HERE/scripts/activate_env.sh"
#
# Why every script does this instead of telling you to activate first.  The scripts are
# meant to be runnable in any order and on their own, and the most common way to get a
# confusing failure out of them is to run one from a shell where the environment is not
# active -- you get `tritonparseoss: command not found`, or worse, a *different* torch.
# Rather than rely on the reader having followed an instruction three sections earlier,
# each script activates the environment itself.
#
# It is deliberately quiet when there is nothing to do, and it never overrides a choice
# you have already made:
#
#   * already inside the artifact's environment  -> nothing happens
#   * a conda environment built by setup_env.sh  -> `conda activate` it
#   * no such environment (a virtualenv, say)    -> leave the caller's PATH alone
#
# Set TRITONPARSE_AE_NO_ACTIVATE=1 to switch it off entirely and use exactly the
# interpreter that is already on PATH.

if [[ "${TRITONPARSE_AE_NO_ACTIVATE:-0}" != "1" ]]; then
    _ae_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    _ae_prefix=""
    [[ -f "$_ae_dir/.env-prefix" ]] && _ae_prefix="$(cat "$_ae_dir/.env-prefix" 2>/dev/null)"

    # Fall back to asking conda, in case setup_env.sh ran somewhere else (or the file was
    # cleaned away) but the environment itself is still there.
    if [[ ! -x "${_ae_prefix:-/nonexistent}/bin/python" ]] && command -v conda >/dev/null 2>&1; then
        _ae_prefix="$(conda env list 2>/dev/null \
                      | awk -v n="${AE_ENV_NAME:-tritonparse-ae}" '$1==n {print $NF}' | head -1)"
    fi

    if [[ -n "$_ae_prefix" && -x "$_ae_prefix/bin/python" ]]; then
        if [[ "${CONDA_PREFIX:-}" == "$_ae_prefix" ]]; then
            :                                        # already there
        else
            _ae_root=""
            [[ -f "$_ae_dir/.conda-root" ]] && _ae_root="$(cat "$_ae_dir/.conda-root" 2>/dev/null)"
            [[ -z "$_ae_root" && -n "${CONDA_EXE:-}" ]] && _ae_root="$(dirname "$(dirname "$CONDA_EXE")")"

            if [[ -f "$_ae_root/etc/profile.d/conda.sh" ]]; then
                # A real `conda activate`, so anything that inspects CONDA_PREFIX or the
                # activation hooks sees a properly entered environment.  conda.sh and the
                # activation hooks are not written against `set -u`, and every caller here
                # sets it, so relax it for the duration and put it back.
                _ae_u=0; [[ $- == *u* ]] && _ae_u=1
                set +u
                # shellcheck disable=SC1091
                source "$_ae_root/etc/profile.d/conda.sh"
                conda activate "$_ae_prefix" 2>/dev/null || export PATH="$_ae_prefix/bin:$PATH"
                [[ "$_ae_u" == "1" ]] && set -u
                unset _ae_u
            else
                # No hook to source (conda was never `conda init`-ed, which is exactly
                # what setup_env.sh leaves you with).  PATH is enough for these scripts.
                export PATH="$_ae_prefix/bin:$PATH"
                export CONDA_PREFIX="$_ae_prefix"
            fi
            printf '  \033[2mactivated %s\033[0m\n' "$_ae_prefix"
        fi
    fi
    unset _ae_dir _ae_prefix _ae_root
fi
