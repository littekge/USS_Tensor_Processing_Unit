#include <Arduino.h>
#include <avr/interrupt.h>

namespace {
constexpr uint32_t kSerialBaud = 115200;
constexpr size_t kBufferSize = 256;

volatile uint8_t gBuffer[kBufferSize];
volatile uint8_t gHead = 0;
volatile uint8_t gTail = 0;

void setupSpiSlave()
{
	pinMode(MISO, OUTPUT);
	pinMode(MOSI, INPUT);
	pinMode(SCK, INPUT);
	pinMode(SS, INPUT_PULLUP);

	SPCR = _BV(SPE) | _BV(SPIE);
	SPSR = 0;
	SPDR = 0;
}

bool popByte(uint8_t &value)
{
	const uint8_t head = gHead;
	const uint8_t tail = gTail;

	if (tail == head) return false;

	value = gBuffer[tail];
	gTail = static_cast<uint8_t>((tail + 1U) % kBufferSize);
	return true;
}

void printHexByte(uint8_t value)
{
	if (value < 0x10) {
		Serial.print('0');
	}

	Serial.print(value, HEX);
}
}  // namespace

uint16_t recieveCount = 0;

ISR(SPI_STC_vect)
{
	recieveCount++;
	const uint8_t received = SPDR;
	SPDR = received; // echo back the received byte as an ACK

	const uint8_t nextHead = static_cast<uint8_t>((gHead + 1U) % kBufferSize);
	if (nextHead != gTail) {
		gBuffer[gHead] = received;
		gHead = nextHead;
	}
}

void setup()
{
	Serial.begin(kSerialBaud);
	setupSpiSlave();

	Serial.println(F("SPI slave ready"));
}

void loop()
{
		uint8_t received = 0;
		while (popByte(received)) {
			Serial.print(F("SPI RX: 0x"));
			printHexByte(received);
			Serial.print(F(" '"));
			if (received >= 32 && received <= 126) Serial.write(received);
			else Serial.print('.');
			Serial.print('\'');
			Serial.println(" " + String(recieveCount));
		}

		static unsigned long lastStatus = 0;
		const unsigned long now = millis();
		if (now - lastStatus >= 1000) {
			lastStatus = now;

			// simplified: no status reporting
		}
}

