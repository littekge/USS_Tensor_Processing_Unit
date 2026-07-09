# Path to python virtual environment (REPLACE WITH YOUR PATH)
pypath="$HOME/Git/USS_Tensor_Processing_Unit/USS_TPU.venv"

# activate venv
source "$pypath/bin/activate"

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
