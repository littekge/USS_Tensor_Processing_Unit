# Pipeline driver: train/export (-t), assemble (-a), program the board (-p).
# Paths are derived from this script's location — run it from anywhere.
SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pypath="$(dirname "$SYSTEM_DIR")/USS_TPU.venv"

if [ ! -d "$pypath" ]; then
  echo "ERROR: virtual environment not found at $pypath" >&2
  echo "Create it first: run ./setup.sh from the repository root." >&2
  exit 1
fi

# activate venv
source "$pypath/bin/activate"

# scripts below are invoked relative to System/
cd "$SYSTEM_DIR"

while getopts ":m:r:atp" opt; do
  case $opt in
  m) model="$OPTARG" ;;
  r) run="$OPTARG" ;;
  a) assemble=true ;;
  t) train=true ;;
  p) program=true ;;
  \?) echo "Invalid option: -$OPTARG" ;;
  :) echo "Option -$OPTARG requires an argument" ;;
  esac
done

if [ "$train" = true ]; then
  python ./Neural_Networks/Start.py $model $run -t -e -r
fi

if [ "$assemble" = true ]; then
  python ./Assembler/Assemble.py $model $run
fi

if [ "$program" = true ]; then
  python ./Communication/PC_2_Arduino/Send_2_Arduino.py
fi

deactivate
