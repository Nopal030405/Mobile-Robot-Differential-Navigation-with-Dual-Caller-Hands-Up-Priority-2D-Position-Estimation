import serial
import time
import threading
import re

class ESP32Controller:
    def __init__(self, port='COM5', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self._running = False
        
        # Data encoder (tick)
        self.enc_left = 0
        self.enc_right = 0
        self.has_hardware_encoder = False
        
        # Variabel untuk estimasi odometri/encoder internal
        self._last_cmd_j1 = 0.0
        self._last_cmd_s1 = 0.0
        self._last_cmd_time = time.time()
        self._lock = threading.Lock()

        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2)  # Stabilisasi koneksi
            print(f">> [SERIAL] Berhasil terhubung ke ESP32 pada port {port}")
            self.is_connected = True
            self._running = True
            
            # Thread pembaca serial untuk menerima data encoder dari ESP32
            self._read_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self._read_thread.start()
        except Exception as e:
            print(f">> [WARNING] Gagal terhubung ke ESP32: {e}")
            self.serial_conn = None
            self.is_connected = False

    def _serial_reader_loop(self):
        """Membaca pesan masuk dari ESP32 (misal format: ENC:left,right)."""
        buffer = ""
        while self._running and self.is_connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._parse_incoming_line(line)
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def _parse_incoming_line(self, line):
        """Mem-parsing string feedback seperti 'ENC:150,148' atau 'ENC_L:150,ENC_R:148'."""
        try:
            if line.startswith("ENC:"):
                parts = line[4:].split(',')
                if len(parts) >= 2:
                    with self._lock:
                        self.enc_left = int(float(parts[0]))
                        self.enc_right = int(float(parts[1]))
                        self.has_hardware_encoder = True
            elif "L:" in line and "R:" in line:
                m_l = re.search(r'L:(-?\d+)', line)
                m_r = re.search(r'R:(-?\d+)', line)
                if m_l and m_r:
                    with self._lock:
                        self.enc_left = int(m_l.group(1))
                        self.enc_right = int(m_r.group(1))
                        self.has_hardware_encoder = True
        except Exception:
            pass

    def send_data(self, j1, s1, j2=0.0, s2=0.0):
        now = time.time()
        dt = max(0.001, min(0.5, now - self._last_cmd_time))
        
        with self._lock:
            if not self.has_hardware_encoder:
                if self._last_cmd_j1 > 0.0:
                    if abs(self._last_cmd_s1) > 2.0:
                        rot_rate = 180.0
                        if self._last_cmd_s1 < 0:
                            self.enc_left += int(rot_rate * dt)
                            self.enc_right -= int(rot_rate * dt)
                        else:
                            self.enc_left -= int(rot_rate * dt)
                            self.enc_right += int(rot_rate * dt)
                    else:
                        fwd_rate = 260.0
                        self.enc_left += int(fwd_rate * dt)
                        self.enc_right += int(fwd_rate * dt)

            self._last_cmd_j1 = j1
            self._last_cmd_s1 = s1
            self._last_cmd_time = now

        if self.is_connected and self.serial_conn:
            payload = f"T1:{j1},S1:{s1},T2:{j2},S2:{s2}\n"
            try:
                self.serial_conn.write(payload.encode('utf-8'))
            except Exception as e:
                print(f">> [ERROR] Gagal mengirim data: {e}")

    def get_encoder_ticks(self):
        """Mengambil nilai encoder saat ini (left, right)."""
        now = time.time()
        dt = max(0.001, min(0.5, now - self._last_cmd_time))
        with self._lock:
            if not self.has_hardware_encoder and self._last_cmd_j1 > 0.0:
                if abs(self._last_cmd_s1) > 2.0:
                    rot_rate = 180.0
                    if self._last_cmd_s1 < 0:
                        self.enc_left += int(rot_rate * dt)
                        self.enc_right -= int(rot_rate * dt)
                    else:
                        self.enc_left -= int(rot_rate * dt)
                        self.enc_right += int(rot_rate * dt)
                else:
                    fwd_rate = 260.0
                    self.enc_left += int(fwd_rate * dt)
                    self.enc_right += int(fwd_rate * dt)
                self._last_cmd_time = now
            return self.enc_left, self.enc_right

    def reset_encoders(self):
        with self._lock:
            self.enc_left = 0
            self.enc_right = 0

    def close(self):
        self._running = False
        if self.is_connected and self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            print(">> [SERIAL] Koneksi ESP32 ditutup.")