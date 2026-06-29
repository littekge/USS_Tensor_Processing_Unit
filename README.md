# USS_Tensor_Processing_Unit

This repository contains research on the synthesis flow and hardware code
required to accelerate a Pytorch neural network using a custom TPU architecture
defined in Verilog HDL that runs on an FPGA. This research is endorsed and
funded by Miami University of Ohio and conducted under the Electrical and
Computer Engineering department in the College of Engineering and Computing.

## Folder Hierarchy and Project Log

This is a general overview of the project folder hierarchy and a log of what
parts of the system have been implemented. Only relevant directories are
included; further subdirectories are specific to components of an individual
system; this is a high-level overview of the outermost systems.

### Specifications

The *Specifications* folder contains shared specification sheets, spec sheet
changelogs, and previous versions of spec sheets.

- Functional TPU ISA (v0.2 complete)
- Functional TPU Message Protocol (v0.2 complete)
- Functional TPU Hardware Spec (v0.1.1 complete, v0.2 in progress)

#### Old

The *Old* folder contains old revisions of spec sheets.

### System

The *System* folder contains the entire TPU system, including neural network
training, data conversion, communication, and TPU hardware.

#### Neural_Networks

The *Neural_Networks* folder contains neural network models (defined using
PyTorch) and code used to train, run, and export them.

- Digit Recognition N.N. (LeNet_5) (complete)
- Extremely small N.N. for initial benchmarks (Tiny_NN) (complete)
- Larger N.N. for testing matrix operation partitioning (Bigger_NN) (complete)
- Python scripts (Start.py, Operations.py) to train, run, and export neural
networks in a format that the synth flow can understand (StableHLO .mlir
format) (complete)

#### Assembler

The *Assembler* folder contains the code that assembles a binary compatible with
the TPU hardware starting from a StableHLO MLIR representation. All of the
data conversion is done here, and additional steps in the synth flow simply
forward data using various communication protocols.

- StableHLO to machine code assembler (v0.1 complete, v0.1.1 pending)

#### Communication

The *Communication* folder contains code that reads and forwards a
TPU-compatible binary to the FPGA.

##### Arduino_2_FPGA

The *Arduino_2_FPGA* folder contains an Arduino program to forward data from the
serial receiver on the Arduino to the SPI output pins.

- Serial Receiver on Arduino (complete)
- SPI Transmitter on Arduino (complete)

##### PC_2_Arduino

The *PC_2_Arduino* folder contains a python script that reads data from the
final binary produced by the assembler, opens a serial port, and forwards
the binary stream to the Arduino.

- Serial Transmitter on PC (complete)

#### Tensor_Processing_Unit

The *Tensor_Processing_Unit* folder contains the TPU hardware written in Verilog
HDL.

- Verilog code for TPU (v0.1.1 complete)

### Tests

The *Tests* folder contains code used to verify the functionality of systems in
the project.

### Current Possible Failure Points

- The resistor divider between the Arduino and FPGA used to rectify voltage
levels between the two devices rounds off the corners of the SPI bus waveform.
If issues begin to arise, a proper logic level converter may be needed.
- The SPI_Slave.v file I found on Github explicitly requires a timing constraint
when crossing clock domains, but I do not currently have one. Will probably not
be an issue.
- The compilation pipeline a version of StableHLO that I built locally. A wheel
for the Python bindings is available in the Github releases, but it does not
contain the necessary StableHLO pass to run the program without a local build.
Currently, this makes the compilation flow difficult to recreate on other
machines.
- The data buffer on the Arduino that connects the computer and FPGA has a fixed
length of 256 bytes and will eventually overflow if the serial input rate
exceeds the SPI send rate.

### Potential Long-Term Changes

- Create soldered perfboard for SPI connection to allow for more reliable data
transfer and faster transfer speed. Probably will not be needed.
- Attempt to package StableHLO binary and Python wheel in a single release to
allow for full integration with Google Colab (makes long term access to script
more reliable).
