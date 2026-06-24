from nn_assembler import Convert
from PC_2_Arduino.Send_2_Arduino import send_2_arduino 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("run_name")
    args = parser.parse_args()
    BIN_PATH = Convert(model_name=args.model_name, run_name=args.run_name)
    print(f"Assembly successful! Binary written to {BIN_PATH}")
    send_2_arduino(BIN_PATH)

