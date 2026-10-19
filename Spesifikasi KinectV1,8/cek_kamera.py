import cv2

print("Memindai seluruh port kamera (0 - 10) yang aktif di laptop...")
for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[TERDETEKSI] Kamera aktif pada indeks: {i}")
        cap.release()
print("Pemindaian selesai.")