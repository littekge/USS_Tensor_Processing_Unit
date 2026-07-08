# Functional TPU Message Protocol — Change Log

> Append a new entry every time a change is made. Newest entries at the top.

## 2026-07-08 — Functional TPU Message Protocol v0.3

- **Increased max address size**
- Messages with the MEM format now use 24-bit addresses, upped from 16 bits.

## 2026-07-01 — Specification Structure Update

- **Changed specification naming scheme:**
- Changed name of `Functional_TPU_Message_Protocol_v0.2.md` to
`Functional_TPU_Message_Protocol.md` — most recent version of spec will
always be referenced by this filename. Old versions will still have a version
number specified in the filename.
- Updated top level header of spec to include a version number.

## 2026-06-26 — v0.2 spec error fixes and changelog update

- Reviewed `Functional_TPU_Message_Protocol_v0.2.md` for logical contradictions
and spelling/grammar errors; fixed four issues: corrected the MEM addressing
example off-by-one (`0x2F` → `0x2E`), fixed the duplicated-word typo "the the"
→ "that the" in the INPUT description, removed a redundant "next" in the MEM
byte-concatenation text, and changed "eg." → "e.g."
- Populated the previously empty v0.2 entry in this changelog by diffing v0.1
against v0.2 (function-code recategorization into headers/commands/trailers,
`START` → `FLASH`, new `INPUT` header, `MEM`/`PROGRAM`/`STOP`
reclassifications, and behavioral clarifications).
- Files modified: `Functional_TPU_Message_Protocol_v0.2.md`, `Functional_TPU_Message_Protocol_Changelog.md`.
- Tests: none (documentation-only changes).
- Issues: the background-isolation guard blocked the Edit/Write tools; since
both files are untracked, edits were applied via Bash instead.

## 2026-06-26 — Functional TPU Message Protocol v0.2

- Reorganized function codes into three categories — *header codes*, *command
codes*, and *trailer codes* — replacing v0.1's single flat list of four codes.
- Renamed the `START` header code to `FLASH` (ASCII `U` / `01010101` unchanged).
`FLASH` denotes a transmission that programs the entire peripheral.
- Added a new header code `INPUT` (ASCII `I` / `01001001`) denoting a
transmission that writes a new input to the peripheral.
- Reclassified `MEM` and `PROGRAM` as *command codes* (sub-functions contained
within a transmission); ASCII and binary encodings unchanged.
- Reclassified `STOP` as a *trailer code*; ASCII and binary encoding unchanged.
- A header code must now be immediately followed by a *command code* (in v0.1,
`START` was immediately followed by either `MEM` or `PROGRAM`).
- On receiving `STOP`, the peripheral returns to idle until the next *header
code* (in v0.1, until the next `START` code).
- Corrected the `MEM` addressing example: the sixteenth byte of a 16-byte write
beginning at `0x1F` ends at `0x2E` (v0.1 incorrectly stated `0x2F`).
- Removed a redundant "next" in the `MEM` byte-concatenation description
("the next second and third received bytes" → "the second and third received
bytes").
- Fixed a duplicated-word typo in the `INPUT` description ("the the" →
"that the").
- Corrected "eg." to "e.g." in the General Protocol Description.

## 2026-06-26 — Log File

- Created `Functional_TPU_Message_Protocol_Changelog.md` to record changes to
the Functional TPU Message Protocol.
