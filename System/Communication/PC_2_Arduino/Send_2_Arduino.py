"""PC -> Arduino serial sender with lock-step, checksummed framing.

This link layer only guards the serial hop between the PC and the Arduino. The
payload bytes it carries are the unmodified TPU message stream that the Arduino
forwards over SPI to the FPGA; the Functional TPU Message Protocol is untouched.

Frame on the wire (PC -> Arduino):
    [LEN][payload: LEN bytes][CRC16 hi][CRC16 lo]
    - LEN in 1..CHUNK_SIZE
    - CRC16-CCITT/XMODEM (poly 0x1021, init 0x0000) over [LEN][payload]
Reply (Arduino -> PC): single byte ACK (valid) or NAK (invalid/timeout).

These framing constants MUST stay identical to the Arduino firmware in
Arduino_2_FPGA/src/main.cpp.
"""

from pathlib import Path
import time

# --- Framing constants (keep in lock-step with the Arduino firmware) --------
BAUD_RATE = 115200
CHUNK_SIZE = 128            # max payload bytes per frame
ACK_BYTE = 0x06            # ASCII ACK
NAK_BYTE = 0x15            # ASCII NAK

# The PC's ceiling for one ACK/NAK. Must exceed the Arduino's worst case: its
# own frame timeout (250 ms) plus receive + SPI drain + reply. 1 s is generous.
ACK_TIMEOUT_S = 1.0
MAX_RETRIES = 5            # retransmissions per chunk before aborting


class FramingError(RuntimeError):
    """Raised when a chunk cannot be delivered within the retry budget."""


def crc16_ccitt(data: bytes) -> int:
    """CRC16-CCITT/XMODEM: poly 0x1021, init 0x0000, no reflection."""
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(payload: bytes) -> bytes:
    """Wrap a payload chunk as [LEN][payload][CRC16 hi][CRC16 lo]."""
    length = len(payload)
    if not 1 <= length <= CHUNK_SIZE:
        raise ValueError(f"payload length {length} out of range 1..{CHUNK_SIZE}")
    body = bytes([length]) + payload
    crc = crc16_ccitt(body)
    return body + bytes([(crc >> 8) & 0xFF, crc & 0xFF])


def stream_content(ser, content: bytes, on_progress=None) -> int:
    """Send content to a serial-like object one lock-step frame at a time.

    ``ser`` must provide write(bytes)->int, read(size)->bytes (honouring a
    read timeout), and reset_input_buffer(). Returns the number of payload
    bytes acknowledged. Raises FramingError if any chunk exhausts its retries.
    """
    ack = bytes([ACK_BYTE])
    nak = bytes([NAK_BYTE])
    total = len(content)
    acked = 0

    for offset in range(0, total, CHUNK_SIZE):
        payload = content[offset:offset + CHUNK_SIZE]
        frame = build_frame(payload)

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                # Clear any stale/late reply before retransmitting.
                ser.reset_input_buffer()
            ser.write(frame)
            reply = ser.read(1)

            if reply == ack:
                acked += len(payload)
                if on_progress is not None:
                    on_progress(acked, total)
                break
            # NAK, timeout (b""), or unexpected byte -> retransmit.
        else:
            chunk_index = offset // CHUNK_SIZE
            raise FramingError(
                f"chunk {chunk_index} (offset {offset}, {len(payload)} bytes) "
                f"failed after {MAX_RETRIES} retransmissions; aborting. "
                f"{acked}/{total} bytes were acknowledged before failure."
            )

    return acked


def find_arduino_port():
    import serial.tools.list_ports

    ports = serial.tools.list_ports.comports()
    for port in ports:
        description = port.description.lower()
        manufacturer = (port.manufacturer or "").lower()
        if "arduino" in description or "arduino" in manufacturer:
            print(f"Found Arduino on port: {port.device}")
            print(f"Description: {port.description}")
            return port.device

    print("No Arduino detected automatically.")
    return None


def send_2_arduino(file_path):
    """Open the Arduino serial port and stream a file with framed flow control."""
    import serial

    assert file_path.is_file()
    content = file_path.read_bytes()
    arduino_port = find_arduino_port()
    ser = serial.Serial(
        port=arduino_port,
        baudrate=BAUD_RATE,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=ACK_TIMEOUT_S,
    )
    print("Attempting serial write...")
    time.sleep(2)  # allow the Arduino auto-reset to finish booting
    ser.reset_input_buffer()

    expected = len(content)

    def report(done, total):
        print(f"\r  {done}/{total} bytes acknowledged", end="", flush=True)

    try:
        acked = stream_content(ser, content, on_progress=report)
    except FramingError as error:
        print()  # finish the progress line
        raise SystemExit(f"ERROR: {error}") from error
    finally:
        ser.close()

    print()
    if acked == expected:
        print(f"Serial write success! {acked} bytes acknowledged by the Arduino.")
    else:
        print(
            "WARNING: acknowledged byte count mismatch:\n"
            f" Expected: {expected}\n Acknowledged: {acked}"
        )


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    FILE_PATH = CURRENT_DIR.parent.parent / "Assembler" / "out" / "TRANSMISSION.bin"
    assert FILE_PATH.is_file()
    send_2_arduino(FILE_PATH)
