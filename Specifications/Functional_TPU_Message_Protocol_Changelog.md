# Functional TPU Message Protocol — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-06-26 — Functional TPU Message Protocol v0.2

**Function code reorganization**

- Reorganized function codes into three categories — *header codes*, *command
  codes*, and *trailer codes* — replacing v0.1's single flat list of four codes.
- Renamed the `START` header code to `FLASH` (ASCII `U` / `01010101` unchanged).
  `FLASH` denotes a transmission that programs the entire peripheral.
- Added a new header code `INPUT` (ASCII `I` / `01001001`) denoting a
  transmission that writes a new input to the peripheral.
- Reclassified `MEM` and `PROGRAM` as *command codes* (sub-functions contained
  within a transmission); ASCII and binary encodings unchanged.
- Reclassified `STOP` as a *trailer code*; ASCII and binary encoding unchanged.

**Behavioral clarifications**

- A header code must now be immediately followed by a *command code* (in v0.1,
  `START` was immediately followed by either `MEM` or `PROGRAM`).
- On receiving `STOP`, the peripheral returns to idle until the next *header
  code* (in v0.1, until the next `START` code).

**Editorial corrections**

- Corrected the `MEM` addressing example: the sixteenth byte of a 16-byte write
  beginning at `0x1F` ends at `0x2E` (v0.1 incorrectly stated `0x2F`).
- Removed a redundant "next" in the `MEM` byte-concatenation description
  ("the next second and third received bytes" → "the second and third received
  bytes").
- Fixed a duplicated-word typo in the `INPUT` description ("the the" →
  "that the").
- Corrected "eg." to "e.g." in the General Protocol Description.

## 2026-06-26 — Log File

- Created `Functional_TPU_Message_Protocol_Changelog.md` to record changes to the Functional TPU Message Protocol.
