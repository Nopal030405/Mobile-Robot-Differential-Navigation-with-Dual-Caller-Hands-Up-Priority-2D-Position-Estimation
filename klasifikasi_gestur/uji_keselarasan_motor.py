# ==============================================================================
# uji_keselarasan_motor.py - Program Khusus Uji Keselarasan & Diagnosa Motor Robot
# ==============================================================================
# File ini berdiri sendiri (STANDALONE) di luar sistem utama navigasi.
# Fitur:
#   1. [L] UJI MOTOR KIRI SAJA (Putar motor kiri maju/mundur)
#   2. [R] UJI MOTOR KANAN SAJA (Putar motor kanan maju/mundur)
#   3. [M] UJI MAJU LURUS (Kedua motor serempak)
#   4. Pengukuran encoder & tingkat deviasi roda (%)
#   5. Checklist perkabelan BTS7960
# ==============================================================================

import os
import sys
import time
import serial
import serial.tools.list_ports
import threading
import re

def dapatkan_port_otomatis():
    """Mendeteksi port COM USB ESP32 secara otomatis."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return 'COM9'
    for p in ports:
        desc = (p.description or '').upper()
        if 'CH340' in desc or 'CP210' in desc or 'SERIAL' in desc or 'USB' in desc:
            return p.device
    return ports[0].device


class MotorAlignmentTester:
    def __init__(self, port=None, baudrate=115200):
        if not port:
            port = dapatkan_port_otomatis()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self._running = False
        
        self.enc_left = 0
        self.enc_right = 0
        self.has_hardware_encoder = False
        self._lock = threading.Lock()

        print(f"\n>> Menghubungkan ke ESP32 pada port '{port}' (baudrate {baudrate})...")
        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=0.05)
            self.serial_conn.dtr = False
            self.serial_conn.rts = False
            time.sleep(2.5)  # Tunggu ESP32 selesai booting
            
            if self.serial_conn.in_waiting > 0:
                self.serial_conn.read(self.serial_conn.in_waiting)
                
            self.is_connected = True
            self._running = True
            
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            print(f">> [BERHASIL] Terhubung ke ESP32 pada {port}!")
        except Exception as e:
            print(f">> [ERROR] Gagal membuka port {port}: {e}")
            self.is_connected = False

    def _reader_loop(self):
        """Mendengarkan data telemetry dari ESP32."""
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
                            self._parse_line(line)
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def _parse_line(self, line):
        """Mendeteksi pesan serial seperti ENC:left,right."""
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

    def kirim_raw(self, teks):
        """Mengirim string mentah ke ESP32."""
        if self.is_connected and self.serial_conn:
            try:
                self.serial_conn.write(teks.encode('utf-8'))
            except Exception as e:
                print(f">> [ERROR] Serial: {e}")

    def kirim_navigasi(self, jarak, sudut):
        """Format standar navigasi T1:j,S1:s."""
        self.kirim_raw(f"T1:{jarak:.2f},S1:{sudut:.2f},T2:0.0,S2:0.0\n")

    def stop(self):
        """Hentikan motor seketika."""
        for _ in range(3):
            self.kirim_navigasi(0.0, 0.0)
            self.kirim_raw("TEST_L:0\n")
            self.kirim_raw("TEST_R:0\n")
            time.sleep(0.02)

    def baca_encoder(self):
        with self._lock:
            return self.enc_left, self.enc_right

    def reset_encoder(self):
        with self._lock:
            self.enc_left = 0
            self.enc_right = 0

    def tutup(self):
        self.stop()
        self._running = False
        if self.serial_conn and self.is_connected:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        print(">> [INFO] Koneksi serial ditutup dengan aman.")


# ==============================================================================
# FITUR PENGUJIAN MOTOR TERPISAH
# ==============================================================================

def uji_motor_kiri(tester, durasi=2.0):
    print("\n" + "=" * 65)
    print("   [UJI MOTOR KIRI] - MENGAKTIFKAN MOTOR SEBELAH KIRI")
    print("=" * 65)
    print(f"Pin ESP32 Motor Kiri : GPIO 25 (RPWM) dan GPIO 26 (LPWM)")
    print(f"Durasi Pengujian     : {durasi} detik")
    print("-----------------------------------------------------------------")
    print(">> Pastikan SAKELAR BATERAI 12V SUDAH 'ON'.")
    input(">> Tekan [ENTER] untuk memutar MOTOR KIRI...")

    tester.reset_encoder()
    enc_l_awal, enc_r_awal = tester.baca_encoder()

    print(f">> [RUNNING] Memutar Motor KIRI...")
    t_start = time.time()
    while (time.time() - t_start) < durasi:
        # Kirim sinyal direct dan sinyal navigasi (sudut +15° memutar motor kiri maju)
        tester.kirim_raw("TEST_L:60\n")
        tester.kirim_navigasi(1.5, 15.0)
        time.sleep(0.05)

    tester.stop()
    time.sleep(0.4)

    enc_l_akhir, enc_r_akhir = tester.baca_encoder()
    delta_l = enc_l_akhir - enc_l_awal
    delta_r = enc_r_akhir - enc_r_awal

    print("-" * 65)
    print(f"HASIL PENGUJIAN MOTOR KIRI:")
    print(f"  * Tick Encoder Kiri  : {delta_l} tick")
    print(f"  * Tick Encoder Kanan : {delta_r} tick")
    if abs(delta_l) > 10:
        print(">> STATUS: [MOTOR KIRI BERFUNGSI NORMAL] (Encoder membaca putaran)")
    else:
        print(">> PERHATIAN: Periksa fisik motor kiri, apakah rodanya berputar?")
    print("=" * 65)


def uji_motor_kanan(tester, durasi=2.0):
    print("\n" + "=" * 65)
    print("   [UJI MOTOR KANAN] - MENGAKTIFKAN MOTOR SEBELAH KANAN")
    print("=" * 65)
    print(f"Pin ESP32 Motor Kanan: GPIO 27 (RPWM) dan GPIO 14 (LPWM)")
    print(f"Durasi Pengujian     : {durasi} detik")
    print("-----------------------------------------------------------------")
    print(">> Pastikan SAKELAR BATERAI 12V SUDAH 'ON'.")
    input(">> Tekan [ENTER] untuk memutar MOTOR KANAN...")

    tester.reset_encoder()
    enc_l_awal, enc_r_awal = tester.baca_encoder()

    print(f">> [RUNNING] Memutar Motor KANAN...")
    t_start = time.time()
    while (time.time() - t_start) < durasi:
        # Kirim sinyal direct dan sinyal navigasi (sudut -15° memutar motor kanan maju)
        tester.kirim_raw("TEST_R:60\n")
        tester.kirim_navigasi(1.5, -15.0)
        time.sleep(0.05)

    tester.stop()
    time.sleep(0.4)

    enc_l_akhir, enc_r_akhir = tester.baca_encoder()
    delta_l = enc_l_akhir - enc_l_awal
    delta_r = enc_r_akhir - enc_r_awal

    print("-" * 65)
    print(f"HASIL PENGUJIAN MOTOR KANAN:")
    print(f"  * Tick Encoder Kiri  : {delta_l} tick")
    print(f"  * Tick Encoder Kanan : {delta_r} tick")
    if abs(delta_r) > 10:
        print(">> STATUS: [MOTOR KANAN BERFUNGSI NORMAL] (Encoder membaca putaran)")
    else:
        print(">> PERHATIAN: Periksa fisik motor kanan, apakah rodanya berputar?")
    print("=" * 65)


def uji_maju_kedua_motor(tester, durasi=2.5):
    print("\n" + "=" * 65)
    print("   [UJI KEDUA MOTOR] - MAJU LURUS SEREMPAK")
    print("=" * 65)
    dur_str = input(f">> Durasi jalan lurus dalam detik [default: {durasi}]: ").strip()
    try:
        dur = float(dur_str) if dur_str else durasi
    except ValueError:
        dur = durasi

    input(">> Tekan [ENTER] untuk mulai jalan lurus...")

    tester.reset_encoder()
    enc_l_awal, enc_r_awal = tester.baca_encoder()

    print(f">> [RUNNING] Mengirim sinyal maju serempak ({dur} detik)...")
    t_start = time.time()
    while (time.time() - t_start) < dur:
        tester.kirim_navigasi(1.8, 0.0)
        time.sleep(0.05)

    tester.stop()
    time.sleep(0.4)

    enc_l_akhir, enc_r_akhir = tester.baca_encoder()
    delta_l = enc_l_akhir - enc_l_awal
    delta_r = enc_r_akhir - enc_r_awal
    selisih = abs(delta_l - delta_r)
    max_tick = max(abs(delta_l), abs(delta_r), 1)
    deviasi = (selisih / max_tick) * 100.0 if (delta_l != 0 or delta_r != 0) else 0.0

    print("-" * 65)
    print(f"HASIL MAJU LURUS:")
    print(f"  * Roda Kiri  : {delta_l} tick")
    print(f"  * Roda Kanan : {delta_r} tick")
    print(f"  * Selisih    : {selisih} tick")
    print(f"  * Deviasi    : {deviasi:.2f} %")
    print("-" * 65)

    if delta_l > 0 and delta_r == 0:
        print(">> DIAGNOSA: Motor KIRI aktif, tapi Motor KANAN MATI TOTAL!")
    elif delta_r > 0 and delta_l == 0:
        print(">> DIAGNOSA: Motor KANAN aktif, tapi Motor KIRI MATI TOTAL!")
    elif delta_l == 0 and delta_r == 0:
        print(">> DIAGNOSA: KEDUA MOTOR TIDAK BERGERAK (Cek baterai / sakelar power).")
    else:
        print(">> DIAGNOSA: KEDUA MOTOR BERFUNGSI!")
    print("=" * 65)


def tampilkan_checklist():
    print("\n" + "=" * 65)
    print("   CHECKLIST PENYEBAB SATU MOTOR MATI (BTS7960 & ESP32)")
    print("=" * 65)
    print("Pin ESP32 ke Driver Motor (ProgramEsp.ino):")
    print("  * MOTOR KIRI : GPIO 25 (RPWM), GPIO 26 (LPWM)")
    print("  * MOTOR KANAN: GPIO 27 (RPWM), GPIO 14 (LPWM)")
    print("\n3 Titik Pemeriksaan Utama pada Motor yang Mati:")
    print("  [1] PIN ENABLE (R_EN dan L_EN) PADA DRIVER BTS7960:")
    print("      Setiap papan BTS7960 memiliki pin R_EN dan L_EN.")
    print("      Pin ini WAJIB dijumper/dihubungkan ke tegangan 5V.")
    print("      -> Jika R_EN/L_EN lepas, driver MATI dan motor tidak akan ada arus!")
    print("  [2] KABEL GROUND (GND):")
    print("      GND ESP32 WAJIB terhubung ke pin GND pada kedua driver BTS7960.")
    print("  [3] KABEL TERMINAL MOTOR (M+ dan M-):")
    print("      Pastikan sekrup terminal kabel motor ke driver kencang dan tidak kendor.")
    print("=" * 65)


def main():
    print("=" * 65)
    print("   PROGRAM UJI KESELARASAN & DIAGNOSA MOTOR ROBOT")
    print("=" * 65)

    port_terdeteksi = dapatkan_port_otomatis()
    print(f">> Port COM terdeteksi di laptop: [{port_terdeteksi}]")
    port_input = input(f">> Tekan ENTER untuk memakai [{port_terdeteksi}] atau ketik port lain: ").strip()
    port = port_input if port_input else port_terdeteksi

    tester = MotorAlignmentTester(port=port)
    if not tester.is_connected:
        print("\n>> Gagal terhubung ke ESP32. Periksa kabel USB.")
        return

    try:
        while True:
            print("\n" + "=" * 65)
            print(f"   MENU PENGUJIAN MOTOR (Port Aktif: {tester.port})")
            print("=" * 65)
            print("  L. UJI MOTOR KIRI SAJA  (Hanya memutar motor sebelah kiri)")
            print("  R. UJI MOTOR KANAN SAJA (Hanya memutar motor sebelah kanan)")
            print("  M. UJI KEDUA MOTOR      (Maju lurus serempak)")
            print("  C. CHECKLIST KABEL      (Panduan cek pin & kabel BTS7960)")
            print("  S. STOP MOTOR SEKETIKA")
            print("  Q. Selesai / Keluar")
            print("=" * 65)

            pilihan = input(">> Pilih menu [L/R/M/C/S/Q]: ").strip().upper()

            if pilihan == 'L':
                uji_motor_kiri(tester)
            elif pilihan == 'R':
                uji_motor_kanan(tester)
            elif pilihan == 'M':
                uji_maju_kedua_motor(tester)
            elif pilihan == 'C':
                tampilkan_checklist()
            elif pilihan == 'S':
                tester.stop()
                print(">> Motor dihentikan.")
            elif pilihan == 'Q':
                break
            else:
                print(">> Pilihan tidak dikenali. Ketik L, R, M, C, S, atau Q.")

    except KeyboardInterrupt:
        print("\n>> Dihentikan (Ctrl+C).")
    finally:
        tester.tutup()


if __name__ == "__main__":
    main()
