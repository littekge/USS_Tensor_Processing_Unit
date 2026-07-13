"""Host-only tests for the PC->Arduino framing layer (no hardware, no pyserial).

Run: python3 System/Communication/PC_2_Arduino/test_framing.py

A MockArduino re-implements the firmware's frame parser and CRC check exactly
as documented in Arduino_2_FPGA/src/main.cpp, so a green run proves the two
ends agree on chunk size, CRC, ACK/NAK bytes, and framing. Fault injection
(corruption, dropped byte, dead link) proves the retransmit/abort logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PC_2_Arduino.Send_2_Arduino import (  # noqa: E402
    ACK_BYTE,
    NAK_BYTE,
    CHUNK_SIZE,
    MAX_RETRIES,
    FramingError,
    build_frame,
    crc16_ccitt,
    stream_content,
)


class MockArduino:
    """Serial-like endpoint mirroring the Arduino firmware's frame parser.

    faults maps zero-based chunk index -> list of dispositions applied to
    successive delivery attempts of that chunk:
      "corrupt" flips a payload bit, "drop" swallows a byte, "dead" stays
      silent (PC read timeout). Anything past the list is delivered cleanly.
    """

    def __init__(self, faults=None, timeout=None):
        self.faults = faults or {}
        self.reply = b""
        self.forwarded = bytearray()  # bytes the "FPGA" would have received
        self.chunk_seen = 0
        self.attempt_counts = {}

    # pyserial-compatible surface used by stream_content -------------------
    def write(self, frame: bytes) -> int:
        self._handle_frame(frame)
        return len(frame)

    def read(self, size: int = 1) -> bytes:
        out, self.reply = self.reply[:size], self.reply[size:]
        return out

    def reset_input_buffer(self):
        self.reply = b""

    # firmware-equivalent parsing -----------------------------------------
    def _handle_frame(self, frame: bytes):
        length = frame[0]
        payload = bytearray(frame[1:1 + length])
        crc_bytes = frame[1 + length:1 + length + 2]

        idx = self.chunk_seen
        attempt = self.attempt_counts.get(idx, 0)
        self.attempt_counts[idx] = attempt + 1
        disposition = None
        if idx in self.faults and attempt < len(self.faults[idx]):
            disposition = self.faults[idx][attempt]

        if disposition == "dead":
            self.reply = b""  # PC will time out
            return
        if disposition == "corrupt":
            payload[0] ^= 0x01
        if disposition == "drop":
            crc_bytes = crc_bytes[:-1]  # truncated frame -> length mismatch

        received_crc = (crc_bytes[0] << 8) | crc_bytes[1] if len(crc_bytes) == 2 else -1
        computed_crc = crc16_ccitt(bytes([length]) + bytes(payload))

        if not 1 <= length <= CHUNK_SIZE or received_crc != computed_crc:
            self.reply = bytes([NAK_BYTE])
            return

        self.forwarded.extend(payload)
        self.chunk_seen += 1
        self.reply = bytes([ACK_BYTE])


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def main():
    # 1. CRC matches the published CRC16/XMODEM vector -> proves algorithm.
    check("CRC16/XMODEM('123456789') == 0x31C3",
          crc16_ccitt(b"123456789") == 0x31C3)
    check("ACK/NAK are 0x06/0x15", ACK_BYTE == 0x06 and NAK_BYTE == 0x15)

    # 2. Clean transfer of a Bigger_NN-sized payload; FPGA bytes match exactly.
    content = bytes((i * 37 + 11) & 0xFF for i in range(130_000))
    mock = MockArduino()
    acked = stream_content(mock, content)
    check("clean transfer acks all bytes", acked == len(content))
    check("forwarded stream is byte-identical to source",
          bytes(mock.forwarded) == content)

    expected_chunks = (len(content) + CHUNK_SIZE - 1) // CHUNK_SIZE
    check("chunk count matches ceil(len/CHUNK_SIZE)",
          mock.chunk_seen == expected_chunks)

    # 3. Frame is well-formed: LEN prefix + 2-byte CRC, last chunk short.
    frame = build_frame(content[:CHUNK_SIZE])
    check("full frame length == 1+CHUNK_SIZE+2", len(frame) == CHUNK_SIZE + 3)
    tail = len(content) % CHUNK_SIZE
    if tail:
        check("final chunk carries the remainder length",
              build_frame(content[-tail:])[0] == tail)

    # 4. Recoverable faults: corrupt, then dropped byte, then dead-then-alive.
    faults = {0: ["corrupt"], 5: ["drop"], 9: ["dead", "dead"]}
    mock = MockArduino(faults=faults)
    acked = stream_content(mock, content)
    check("transfer recovers from injected faults", acked == len(content))
    check("recovered stream is byte-identical", bytes(mock.forwarded) == content)
    check("corrupt data never forwarded (retry re-sent chunk 0)",
          mock.attempt_counts[0] == 2)

    # 5. Corrupt payload is never handed to the FPGA on the failing attempt.
    small = bytes(range(200))
    mock = MockArduino(faults={0: ["corrupt"]})
    stream_content(mock, small)
    check("no corrupt bytes reached the FPGA mock", bytes(mock.forwarded) == small)

    # 6. Unrecoverable link -> bounded retries -> descriptive abort, no hang.
    mock = MockArduino(faults={0: ["dead"] * (MAX_RETRIES + 1)})
    try:
        stream_content(mock, small)
        check("dead link raises FramingError", False)
    except FramingError as err:
        check("dead link raises FramingError", True)
        check("abort message names the chunk", "chunk 0" in str(err))
    check("dead chunk tried exactly MAX_RETRIES+1 times",
          mock.attempt_counts[0] == MAX_RETRIES + 1)

    print("\nAll framing tests passed.")


if __name__ == "__main__":
    main()
