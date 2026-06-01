# USS_Tensor_Processing_Unit
This repository contains research on the sythesis flow and hardware code required to accelerate a pytorch neural network using an FPGA. This research is endorsed and funded by Miami University of Ohio and conducted under their Electrical and Computer Engineering department in the College of Engineering and Computing.

# Folder Heirarchy and Project Log:
This is a general overview of the project folder hierarchy and a log of what parts of the system have been implemented. Only relevant directories are included; further subdirectories are either self-explanatory or irrelevent to the overall function of the system. 
## System
The system folder contains the entire TPU system, including data conversion, communication, and TPU hardware.
### Neural_Networks
The *Neural_Networks* folder contains neural network models (defined using PyTorch) and outermost application code.
- Digit Recognition N.N. (LeNet-5) (complete)
- Extremely small N.N. for initial benchmarks (TinyNN) (complete)
### Synth_Flow
The *Synth_Flow* folder contains code to translate a neural network into a data format usable by the TPU and subsequently load it onto an FPGA.
#### Neural_Network_2_Serial
The *NeuralNetwork2Serial* folder contains an MLIR oipeline that converts a neural network into serial data formatted for the FPGA. All of the data conversion is done here, and additional steps in the synth flow simply forward data using various communication protocoals.
- MLIR conversion (in progress)
#### Arduino_2_FPGA
The *Arduino_2_FPGA* folder contains an Arduino program to forward data from the serial reciever on the arduino to the SPI output pins.
- Serial Reciever on Arduino (complete)
- SPI Transmitter on Arduino (complete)
### Tensor_Processing_Unit
The *Tensor_Processing_Unit* folder contains the TPU hardware written in Verilog HDL. 
#### TPU_VGA_DEBUG
The *TPU_VGA_DEBUG* folder contains debug code that can output numbers to a monitor through VGA.
- debug module (complete)
- VGA controller (complete)
#### TPU_SPI_INTERFACE
The *TPU_SPI_INTERFACE* folder contains the SPI reciever and buffer that communicate with the Arduino.
- SPI reciever module (complete)
- SPI recieve buffer (complete)
## Tests
The *Tests* folder contains code used to verify the functionality of systems in the project.

## Current Possible Failure Points
1. The resistor divider between the Arduino and FPGA used to rectify voltage levels between the two devices rounds off the corners of the SPI bus waveform. If issues begin to arise, a proper logic level converter may be needed.
2. The SPI_Slave.v file I found on Github explicitly requires a timing constraint when crossing clock domains, but I do not currently have one. 
