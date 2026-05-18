import serial
import time

ser = serial.Serial(
port = 'COM6',
baudrate = 115200,
parity = serial.PARITY_NONE,
stopbits = serial.STOPBITS_ONE,
bytesize = serial.EIGHTBITS,
timeout = 1
)
time.sleep(3)
ser.reset_input_buffer()

for k in range(10):
    for i in range(99):
        ser.write(bytes([i]))
    time.sleep(0.2)
    
    


"""
strtx = "this is a very long test string"
ser.write(strtx.encode())

chartx = ['T', 'E', 'S', 'T']

for k in chartx:
    ser.write(bytes(k, "utf-8"))
    ser.flush()

tx = [0b01010101, 0xFF, 0x23, 0x03, 0x29]
for x in tx:
    ser.write(bytes([x]))
    ser.flush()
"""
ser.close()

