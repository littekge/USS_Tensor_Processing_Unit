# Benchmarks

> This folder contains 3 example transmissions used to program the TPU. They are
> saved here for the purpose of skipping the compilation pipeline if it's too
> difficult to set up.

## Transmission format

Each transmission is formatted according to the *Functional TPU Message
Protocol* (See `/Specifications`). They consist of two sections:

1. Static weight values
2. Program instructions

The static weight values are ALWAYS transmitted first. Transmissions are stored
and read from the first byte to the last. Program instructions are from the
*Functional TPU ISA*.

## Tiny_NN

**Description:** Very small, fully linear neural network. Used as a bare-minimum benchmark.
**Input:** Single value
**Expected Output:** Output = Input - 2

## Bigger_NN

**Description:** Much larger version of Tiny_NN. Used to test large neural
networks and transmission reliability.
**Input:** Single value
**Expected Output:** Output = Input - 2

## LeNet_5

**Description:** Full convolutional neural network for digit recognition.
**Input:** 28x28 greyscale image of a handwritten number.
**Expected Output:** 10 values indexed 0-9. Each value corresponds to the
probability that the input image is the number at that index (e.g. if a picture
of the number 8 is the input, the output value at index 8 should be the highest).
