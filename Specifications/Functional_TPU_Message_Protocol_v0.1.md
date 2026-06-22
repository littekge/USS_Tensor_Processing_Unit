# Functional TPU Message Protocol
> **Purpose:** This document outlines the *Functional TPU* messaging protocol.

## General Protocol Description
The *Functional TPU* messaging protocol is a controller-peripheral protocol that defines the format of bytes transmitted over a lower level communication protocol (eg. SPI or UART). The protocol is simplex, meaning that it defines data flow in only a single direction (controller to peripheral). The protocol uses single-byte *function codes* to indicate how the peripheral should interpret received data. All data transmissions must be preceded by a function code.

### Byte Order
For any function code that indicates the transmission of data larger than a single byte, the data will be sent with the least significant byte first.

## Function Codes

| Function Code | ASCII | Binary
| --- | --- | --- |
| START | U | 01010101 |
| STOP | S | 01010011 |
| MEM | M | 01001101 |
| PROGRAM | P | 01010000 |

The *Functional TPU* messaging protocol defines 4 function codes (START, MEM, PROGRAM, STOP). All function codes are a single byte ASCII code. Each code defines how the peripheral interprets the next N received bytes. Function code descriptions describe the order in which the bytes are received from left to right.

### START
The START function code indicates the beginning of a transmission. The peripheral should disregard all received bytes until it receives the START code. The START code should be immediately followed by either a MEM or PROGRAM code.

### MEM

| Function Code | Lower Address | Upper Address | Lower Length | Upper Length | Data | ... |
| --- | --- | --- | --- | --- | --- | --- |
| MEM | LADD | UADD | LLEN | ULEN | DATA | ... |

The MEM function code indicates that the following data should be written to device memory. Upon receiving the MEM function code during a transmission, the peripheral should concatenate the next second and third received bytes into a 16-bit address where LADD is the bottom 8 bits and UADD is the upper 8 bits. Similarly, the peripheral should concatenate the fourth and fifth received bytes into a 16-bit length value where LLEN is the bottom 8 bits and ULEN is the upper 8 bits. The length value indicates how many subsequently received DATA bytes should be written to memory starting from the initial 16-bit address. For example, if 16 bytes need to be written to memory address 0x1F, the first DATA byte should be written to 0x1F, the second to 0x20, the third to 0x21, and so on until the sixteenth byte is written to 0x2F. Note that memory addresses greater than 2^16-1=65535 are not currently supported.

### PROGRAM

| Function Code | Instruction Byte 0 | Instruction Byte 1 | ... | Instruction Byte 15 |
| --- | --- | --- | --- | --- |
| PROGRAM | IB0 | IB1 | ... | IB15

The PROGRAM function code indicates that the following data should be written to program memory as an *instruction*. Upon receiving the PROGRAM function code during a transmission, the peripheral should concatenate the next 16 received bytes into a 128-bit instruction (LSB received first (see *Byte Order*)) and subsequently write the instruction to the next available location in program memory. 

### STOP
The STOP function code indicates the end of a transmission. When the peripheral receives the STOP code, it returns to an idle state until it receives the next START code. 

## Invalid Function Codes
In the event that the peripheral expects a function code but receives another value the peripheral may define an internal ERROR state and the communication channel is assumed to be compromised. If an internal ERROR state is defined, the peripheral must also define how the ERROR state is reset so that communication may resume or begin again.