#include <Arduino.h>
#include <SPI.h>

constexpr unsigned long SERIAL_BAUD_RATE = 115200;
constexpr uint32_t SPI_CLOCK_HZ = 8000000UL;
constexpr uint8_t FPGA_CS_PIN = SS;
constexpr size_t BUFFER_SIZE = 256;
constexpr size_t CHUNK_SIZE = 32;

struct ByteRingBuffer {
  uint8_t data[BUFFER_SIZE];
  uint8_t head = 0;
  uint8_t tail = 0;

  bool empty() const {
    return head == tail;
  }

  bool full() const {
    return static_cast<uint8_t>(head + 1U) == tail;
  }

  bool push(uint8_t value) {
    const uint8_t nextHead = static_cast<uint8_t>(head + 1U);
    if (nextHead == tail) {
      return false;
    }

    data[head] = value;
    head = nextHead;
    return true;
  }

  uint8_t pop() {
    const uint8_t value = data[tail];
    tail = static_cast<uint8_t>(tail + 1U);
    return value;
  }

  uint8_t available() const {
    return static_cast<uint8_t>(head - tail);
  }
};

ByteRingBuffer serialBuffer;
SPISettings spiSettings(SPI_CLOCK_HZ, MSBFIRST, SPI_MODE0);

unsigned long bytesReceived = 0;
unsigned long bytesForwarded = 0;
unsigned long bufferWaits = 0;

void drainBufferToSpi() {
  if (serialBuffer.empty()) {
    return;
  }

  uint8_t chunk[CHUNK_SIZE];

  SPI.beginTransaction(spiSettings);
  digitalWrite(FPGA_CS_PIN, LOW);

  while (!serialBuffer.empty()) {
    const uint8_t chunkLength = serialBuffer.available() < CHUNK_SIZE
      ? serialBuffer.available()
      : CHUNK_SIZE;

    for (uint8_t index = 0; index < chunkLength; ++index) {
      chunk[index] = serialBuffer.pop();
    }

    SPI.transfer(chunk, chunkLength);
    bytesForwarded += chunkLength;
  }

  digitalWrite(FPGA_CS_PIN, HIGH);
  SPI.endTransaction();
}

bool serialToSpi() {
  bool movedBytes = false;

  while (Serial.available() > 0) {
    if (serialBuffer.full()) {
      ++bufferWaits;
      drainBufferToSpi();
      movedBytes = true;
      continue;
    }

    const int incoming = Serial.read();
    if (incoming < 0) {
      break;
    }

    if (serialBuffer.push(static_cast<uint8_t>(incoming))) {
      ++bytesReceived;
    }
  }

  if (!serialBuffer.empty()) {
    movedBytes = true;
  }

  drainBufferToSpi();
  return movedBytes;
}

//#define DEBUG_SERIAL

void setup() {
  Serial.begin(115200);
  pinMode(FPGA_CS_PIN, OUTPUT);
  digitalWrite(FPGA_CS_PIN, HIGH);
  SPI.begin();
}

void loop() {
  
  #ifdef DEBUG_SERIAL
  while (Serial.available() > 0) {
    uint8_t byte = Serial.read();
    Serial.write(byte);
  }
  #else
  serialToSpi();
  #endif
}