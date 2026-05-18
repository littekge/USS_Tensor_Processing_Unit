import serial
import serial.tools.list_ports
import time


ser = serial.Serial(
port = 'COM3',
baudrate = 115200,
parity = serial.PARITY_NONE,
stopbits = serial.STOPBITS_ONE,
bytesize = serial.EIGHTBITS,
timeout = 1
)
time.sleep(3)
ser.reset_input_buffer()


tx = [0b01010101, 0xFF, 0x23, 0x03, 0x29]
for x in tx:
    ser.write(bytes([x]))
    ser.flush()
    rx = ser.read(1)
    print("rx raw:", rx)
    print("rx int:", rx[0])
    print("rx bin:", format(rx[0], "08b"))
ser.close()

