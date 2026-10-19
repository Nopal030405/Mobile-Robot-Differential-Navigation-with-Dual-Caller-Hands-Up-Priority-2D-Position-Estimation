// ==========================================
// DEFINISI PIN DRIVER BTS7960
// ==========================================
#define RPWM_L  25
#define LPWM_L  26
#define RPWM_R  27
#define LPWM_R  14

// ==========================================
// DEFINISI PIN ENCODER PG28
// ==========================================
#define ENC_L_A 18  
#define ENC_L_B 19  
#define ENC_R_A 32  
#define ENC_R_B 33  

const int pwmFreq = 10000;    
const int pwmResolution = 8;  

volatile long tick_count_left = 0;
volatile long tick_count_right = 0;

float jarak1 = 0.0, sudut1 = 0.0;
unsigned long lastDataTime = 0;
int currentSpeed = 0; // Kecepatan persisten antar iterasi loop()

void IRAM_ATTR isr_encoder_left() {
  if (digitalRead(ENC_L_B) == HIGH) tick_count_left--;
  else tick_count_left++;
}

void IRAM_ATTR isr_encoder_right() {
  if (digitalRead(ENC_R_B) == HIGH) tick_count_right++;
  else tick_count_right--;
}

// Fungsi dengan arah putaran yang sudah tervalidasi benar
void driveMotors(int speedL, int speedR) {
  // Motor Kiri
  if (speedL > 0) {
    ledcWrite(RPWM_L, abs(speedL)); ledcWrite(LPWM_L, 0);
  } else if (speedL < 0) {
    ledcWrite(RPWM_L, 0); ledcWrite(LPWM_L, abs(speedL));
  } else {
    ledcWrite(RPWM_L, 0); ledcWrite(LPWM_L, 0);
  }

  // Motor Kanan
  if (speedR > 0) {
    ledcWrite(RPWM_R, 0); ledcWrite(LPWM_R, abs(speedR));
  } else if (speedR < 0) {
    ledcWrite(RPWM_R, abs(speedR)); ledcWrite(LPWM_R, 0);
  } else {
    ledcWrite(RPWM_R, 0); ledcWrite(LPWM_R, 0);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  pinMode(ENC_L_A, INPUT_PULLUP); pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP); pinMode(ENC_R_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isr_encoder_left, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isr_encoder_right, RISING);

  ledcAttach(RPWM_L, pwmFreq, pwmResolution); ledcAttach(LPWM_L, pwmFreq, pwmResolution);
  ledcAttach(RPWM_R, pwmFreq, pwmResolution); ledcAttach(LPWM_R, pwmFreq, pwmResolution);

  driveMotors(0, 0);
}

String getValue(String data, char separator, int index) {
  int found = 0;
  int strIndex[] = {0, -1};
  int maxIndex = data.length() - 1;
  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == separator || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void loop() {
  // 1. BACA DATA DARI PYTHON
  if (Serial.available() > 0) {
    String incomingData = Serial.readStringUntil('\n');
    incomingData.trim();
    
    if (incomingData.startsWith("T1:")) {
      jarak1 = getValue(incomingData, ',', 0).substring(3).toFloat();
      sudut1 = getValue(incomingData, ',', 1).substring(3).toFloat();
      lastDataTime = millis();
    }
  }

  // 1b. KIRIM TELEMETRI ENCODER KE PYTHON (TIAP 100MS)
  static unsigned long lastEncSend = 0;
  if (millis() - lastEncSend >= 100) {
    lastEncSend = millis();
    Serial.print("ENC:");
    Serial.print(tick_count_left);
    Serial.print(",");
    Serial.println(tick_count_right);
  }

  // 2. SAFETY STOP MUTLAK
  if (jarak1 <= 0.0 || (millis() - lastDataTime > 2000 && lastDataTime != 0)) {
    currentSpeed = 0;
    driveMotors(0, 0);
    return; 
  }

  // 3. LOGIKA NAVIGASI (ALIGN-FIRST + RAMP UP/DOWN)
  float batasAman = 0.7; 
  int targetSpeed = 60;  
  int speedPutar = 50; // Kecepatan untuk berputar mencari titik tengah

  if (jarak1 > batasAman) {
    
    // --- FASE A: PENYELARASAN SUDUT (CARI TITIK TENGAH) ---
    if (sudut1 > 3.0) {
      currentSpeed = 0; // Reset momentum maju
      driveMotors(speedPutar, -speedPutar); // Putar Kanan
      delay(20);
      return; 
    } 
    else if (sudut1 < -3.0) {
      currentSpeed = 0; // Reset momentum maju
      driveMotors(-speedPutar, speedPutar); // Putar Kiri
      delay(20);
      return; 
    }

    // --- FASE B: MAJU LURUS (KARENA SUDUT SUDAH AMAN DI -3 HINGGA 3) ---
    if (jarak1 < 1.3) {
      // RAMP-DOWN: Melambat halus
      float span = 1.3 - batasAman;
      float currentSpan = jarak1 - batasAman;
      currentSpeed = map(currentSpan * 100, 0, span * 100, 25, targetSpeed);
      currentSpeed = constrain(currentSpeed, 25, targetSpeed);
    } else {
      // RAMP-UP: Akselerasi halus
      if (currentSpeed < targetSpeed) {
        currentSpeed += 2; // Penambahan step pelan
        if (currentSpeed < 25) currentSpeed = 25; // Tenaga awal agar motor tidak berdengung
      }
      currentSpeed = constrain(currentSpeed, 25, targetSpeed);
    }

    // Eksekusi maju lurus
    driveMotors(currentSpeed, currentSpeed);
    delay(20); // Pacing waktu agar ramp-up tidak terjadi secara instan

  } else {
    // 4. MENCAPAI FINISH POINT
    currentSpeed = 0;
    driveMotors(0, 0);
  }
}
