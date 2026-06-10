# USS_Tensor_Processing_Unit
This repository contains research on the synthesis flow and hardware code required to accelerate a Pytorch neural network using a custom TPU architecture defined in Verilog HDL that runs on an FPGA. This research is endorsed and funded by Miami University of Ohio and conducted under their Electrical and Computer Engineering department in the College of Engineering and Computing.

# Folder Hierarchy and Project Log:
This is a general overview of the project folder hierarchy and a log of what parts of the system have been implemented. Only relevant directories are included; further subdirectories are specific to components of an individual system; this is a high-level overview of the outermost systems.
## System
The system folder contains the entire TPU system, including data conversion, communication, and TPU hardware.
### Neural_Networks
The *Neural_Networks* folder contains neural network models (defined using PyTorch) and outermost application code.
- Digit Recognition N.N. (LeNet_5) (complete)
- Extremely small N.N. for initial benchmarks (Tiny_NN) (complete)
- Larger N.N. for testing matrix operation partitioning (Bigger_NN) (complete)
- Python scripts (Start.py, Operations.py) to train, run, and export neural networks in a format that the synth flow can understand (StableHLO .mlir format) (complete)
### Synth_Flow
The *Synth_Flow* folder contains code to translate a neural network into a data format usable by the TPU and subsequently load it onto an FPGA.
#### Neural_Network_2_Serial
The *Neural_Network_2_Serial* folder contains an MLIR pipeline that converts a neural network into serial data formatted for the FPGA. All of the data conversion is done here, and additional steps in the synth flow simply forward data using various communication protocols.
- StableHLO optimization script (convert.py) (complete)
- StableHLO to machine code assembler (pending)
#### Arduino_2_FPGA
The *Arduino_2_FPGA* folder contains an Arduino program to forward data from the serial receiver on the Arduino to the SPI output pins.
- Serial Receiver on Arduino (complete)
- SPI Transmitter on Arduino (complete)
### Tensor_Processing_Unit
The *Tensor_Processing_Unit* folder contains the TPU hardware written in Verilog HDL.
- Functional TPU ISA (v0.2 complete)
- Message protocol (v0.1 complete)
- Hardware Spec (in progress)
- Verilog code for TPU (pending)
## Tests
The *Tests* folder contains code used to verify the functionality of systems in the project.

## Current Possible Failure Points
- The resistor divider between the Arduino and FPGA used to rectify voltage levels between the two devices rounds off the corners of the SPI bus waveform. If issues begin to arise, a proper logic level converter may be needed.
- The SPI_Slave.v file I found on Github explicitly requires a timing constraint when crossing clock domains, but I do not currently have one. 
- The compilation pipeline a version of StableHLO that I built locally. A wheel for the Python bindings is available in the Github releases, but it does not contain the necessary StableHLO pass to run the program without a local build. Currently, this makes the compilation flow difficult to recreate on other machines.
- The data buffer on the Arduino that connects the computer and FPGA has a fixed length of 256 bytes and could overflow if the serial input rate greatly exceeds the SPI send rate.

## Potential Long-Term Changes
- Create soldered perfboard for SPI connection to allow for more reliable data transfer and faster transfer speed.
- Attempt to package StableHLO binary and Python wheel in a single release to allow for full integration with Google Colab (makes long term access to script more reliable).
- Create custom MLIR passes that handle partitioning of oversize operations.
