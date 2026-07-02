from pathlib import Path
import serial
import serial.tools.list_ports
import time

def find_arduino_port():
    # Retrieve all available serial ports
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Check descriptions or manufacturer metadata for 'arduino'
        description = port.description.lower()
        manufacturer = (port.manufacturer or "").lower()
        
        if "arduino" in description or "arduino" in manufacturer:
            print(f"Found Arduino on port: {port.device}")
            print(f"Description: {port.description}")
            return port.device
            
    print("No Arduino detected automatically.")
    return None

# forwards bytes directly from a file to a serial port
def send_2_arduino(file_path):
    assert file_path.is_file()
    content = file_path.read_bytes()
    arduino_port = find_arduino_port()
    ser = serial.Serial(
        port=arduino_port,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,
        timeout = 1
    )
    print("Attempting serial write...")
    time.sleep(2)
    ser.reset_input_buffer()
    expected = len(content);
    sent = ser.write(content)
    if (sent == expected):
        print(f"Serial write success! {sent} bytes sent over serial.")
    else:
        print(f"WARNING: mismatch in expected number of sent bytes: \n Expected: {expected} \n Sent: {sent}")
    
if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).parent
    FILE_PATH = CURRENT_DIR.parent / "Neural_Network_2_TPU_ISA" / "out" / "TRANSMISSION.bin"
    assert FILE_PATH.is_file()
    send_2_arduino(FILE_PATH)


