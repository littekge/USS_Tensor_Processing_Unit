#include <Arduino.h>
#include <SPI.h>

// ---------------------------------------------------------------------------
// PC <-> Arduino link-layer framing (must stay identical to the PC sender in
// PC_2_Arduino/Send_2_Arduino.py). This layer only guards the serial hop; the
// bytes forwarded over SPI to the FPGA are the unmodified TPU message stream.
//
// Frame on the wire (PC -> Arduino):
//   [LEN][payload: LEN bytes][CRC16 hi][CRC16 lo]
//   - LEN in 1..CHUNK_SIZE
//   - CRC16-CCITT/XMODEM (poly 0x1021, init 0x0000) over [LEN][payload]
// Reply (Arduino -> PC): single byte ACK (valid) or NAK (invalid/timeout).
//
// Lock-step: the PC sends exactly one frame, then blocks for ACK/NAK. The
// Arduino receives the whole frame, verifies it, forwards payload over SPI,
// and only then replies. No serial arrives while SPI drains, so the UART FIFO
// cannot overrun -- this is what removes the dropped-byte failure at 115200.
// ---------------------------------------------------------------------------

constexpr unsigned long SERIAL_BAUD_RATE = 115200;
constexpr uint32_t SPI_CLOCK_HZ = 125000UL;
constexpr uint8_t FPGA_CS_PIN = SS;

// Preserved 256-byte buffer budget (see project notes: buffer sizing is a
// user decision). It now holds one lock-step frame: [LEN][payload][CRC16].
constexpr size_t BUFFER_SIZE = 256;
constexpr uint8_t CHUNK_SIZE = 128;   // max payload bytes per frame

constexpr uint8_t ACK_BYTE = 0x06;    // ASCII ACK
constexpr uint8_t NAK_BYTE = 0x15;    // ASCII NAK

// Once a frame has started (LEN read), every remaining byte must arrive within
// this window; otherwise the frame is treated as lost. A full 131-byte frame
// at 115200 baud takes ~11 ms, so this is generous headroom.
constexpr unsigned long FRAME_TIMEOUT_MS = 250;
// Quiet-line gap used to drain stale/garbled bytes before replying NAK.
constexpr unsigned long FLUSH_IDLE_MS = 50;

SPISettings spiSettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE0);

uint8_t frameBuffer[BUFFER_SIZE];

// Diagnostic counters. Not printed during normal operation: the serial link is
// now a binary ACK/NAK channel, so text would corrupt the protocol. Inspect
// via debugger or a dedicated DEBUG build.
unsigned long framesAccepted = 0;
unsigned long framesRejected = 0;
unsigned long bytesReceived = 0;
unsigned long bytesForwarded = 0;

uint16_t crc16Ccitt(const uint8_t* data, size_t length) {
  uint16_t crc = 0x0000;
  for (size_t i = 0; i < length; ++i) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (uint8_t bit = 0; bit < 8; ++bit) {
      if (crc & 0x8000) {
        crc = static_cast<uint16_t>((crc << 1) ^ 0x1021);
      } else {
        crc = static_cast<uint16_t>(crc << 1);
      }
    }
  }
  return crc;
}

bool readByteTimed(uint8_t& out, unsigned long timeoutMs) {
  const unsigned long start = millis();
  while (Serial.available() == 0) {
    if (millis() - start > timeoutMs) {
      return false;
    }
  }
  out = static_cast<uint8_t>(Serial.read());
  return true;
}

void flushSerialRx() {
  unsigned long lastByte = millis();
  while (millis() - lastByte < FLUSH_IDLE_MS) {
    if (Serial.available() > 0) {
      Serial.read();
      lastByte = millis();
    }
  }
}

void forwardToSpi(const uint8_t* data, uint8_t length) {
  SPI.beginTransaction(spiSettings);
  digitalWrite(FPGA_CS_PIN, LOW);
  for (uint8_t i = 0; i < length; ++i) {
    SPI.transfer(data[i]);
  }
  digitalWrite(FPGA_CS_PIN, HIGH);
  SPI.endTransaction();
  bytesForwarded += length;
}

void rejectFrame() {
  flushSerialRx();
  ++framesRejected;
  Serial.write(NAK_BYTE);
}

void receiveFrame() {
  // Idle until a frame begins; LEN has no timeout so the link can sit quiet.
  if (Serial.available() == 0) {
    return;
  }

  const uint8_t length = static_cast<uint8_t>(Serial.read());
  if (length == 0 || length > CHUNK_SIZE) {
    rejectFrame();  // desync/garbage LEN
    return;
  }

  frameBuffer[0] = length;

  // payload followed by 2 CRC bytes
  const uint16_t remaining = static_cast<uint16_t>(length) + 2U;
  for (uint16_t i = 0; i < remaining; ++i) {
    uint8_t incoming;
    if (!readByteTimed(incoming, FRAME_TIMEOUT_MS)) {
      rejectFrame();  // dropped byte -> frame never completes
      return;
    }
    frameBuffer[1 + i] = incoming;
  }

  const uint16_t receivedCrc =
    (static_cast<uint16_t>(frameBuffer[1 + length]) << 8) |
    static_cast<uint16_t>(frameBuffer[1 + length + 1]);
  const uint16_t computedCrc = crc16Ccitt(frameBuffer, static_cast<size_t>(length) + 1U);

  if (receivedCrc != computedCrc) {
    rejectFrame();  // corruption -> never forwarded to FPGA
    return;
  }

  bytesReceived += length;
  forwardToSpi(&frameBuffer[1], length);
  ++framesAccepted;
  Serial.write(ACK_BYTE);
}

//#define DEBUG_SERIAL

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  pinMode(FPGA_CS_PIN, OUTPUT);
  digitalWrite(FPGA_CS_PIN, HIGH);
  SPI.begin();
}

void loop() {
#ifdef DEBUG_SERIAL
  // Raw loopback for link bring-up: echoes bytes without framing.
  while (Serial.available() > 0) {
    const uint8_t value = static_cast<uint8_t>(Serial.read());
    Serial.write(value);
  }
#else
  receiveFrame();
#endif
}
