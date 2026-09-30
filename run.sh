#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# CPU-only PyTorch. The default PyPI wheel bundles CUDA, which would add
# several GB of nvidia-* packages that are never used here.
CPU_TORCH_INDEX="https://download.pytorch.org/whl/cpu"

needs_streamlit() {
  local arg
  for arg in "$@"; do
    [ "$arg" = "streamlit" ] && return 0
  done
  case " $* " in
    *" -m streamlit "*) return 0 ;;
  esac
  return 1
}

if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
fi

if [ ! -x .venv/bin/streamlit ] && needs_streamlit "$@"; then
  REQS=requirements-ui.txt
elif [ -f requirements-core.stamp ]; then
  REQS=""
elif [ -x .venv/bin/python ] && .venv/bin/python -c "import langchain_community" 2>/dev/null; then
  REQS=""
else
  REQS=requirements.txt
fi

if [ -n "$REQS" ]; then
  echo "Installing dependencies from $REQS (first run takes a few minutes)..."
  .venv/bin/pip install --quiet torch --index-url "$CPU_TORCH_INDEX"
  .venv/bin/pip install --quiet -r "$REQS"
  [ "$REQS" = "requirements.txt" ] && touch requirements-core.stamp
fi

exec .venv/bin/python "$@"
