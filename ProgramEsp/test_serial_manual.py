import time
from Esp32.send_to_esp32 import ESP32Controller

print("="*60)
print("     TEST MANUAL SERIAL KE ESP32 & MOTOR")
print("="*60)

# Inisialisasi koneksi ke ESP32 (Sesuaikan 'COM3' dengan port Anda)
esp32 = ESP32Controller(port='COM5', baudrate=115200)

if not esp32.is_connected:
    print(">> [GAGAL] ESP32 tidak terhubung. Periksa kabel USB atau port COM.")
    exit()

print("\n>> Koneksi berhasil! Ketik perintah uji coba di bawah ini.")
print("   Contoh format: T1:1.5,S1:15.0,T2:2.0,S2:-10.0")
print("   Ketik 'exit' untuk keluar.\n")

while True:
    try:
        # Input manual dari terminal
        user_input = input("Masukkan data (Format T1:j,S1:s,T2:j,S2:s): ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        if not user_input:
            continue
            
        # Kirim string mentah langsung ke ESP32
        payload = user_input + "\n"
        esp32.serial_conn.write(payload.encode('utf-8'))
        print(f">> [TERKIRIM]: {user_input}")
        
        # Tunggu sebentar untuk melihat respon serial monitor jika ada
        time.sleep(0.5)
        
    except KeyboardInterrupt:
        break

esp32.close()
print(">> Tes manual selesai.")