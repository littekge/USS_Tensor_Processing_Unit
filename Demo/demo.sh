# LeNet-5 live demo launcher. Paths are derived from this script's location.
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pypath="$(dirname "$DEMO_DIR")/USS_TPU.venv"

if [ ! -d "$pypath" ]; then
  echo "ERROR: virtual environment not found at $pypath" >&2
  echo "Create it first: run ./setup.sh from the repository root." >&2
  exit 1
fi

source "$pypath/bin/activate"       # activate venv
python "$DEMO_DIR/LeNet5_Demo.py"   # run script
deactivate
