#include <Arduino.h>


#define MCP_TX_PIN  17  
#define MCP_RX_PIN  16  
#define MCP_EN_PIN  33  

// LEDs
#define PIN_SPEED 14
#define PIN_STOP  12

uint8_t vitesseRecue = 0;
bool obstacleDetecte = false;
unsigned long dernierSignalMillis = 0;

void setup() {
  Serial.begin(9600);   // Debug
  Serial2.begin(9600, SERIAL_8N1, MCP_RX_PIN, MCP_TX_PIN);
  
 
  
  // Enable du MCP2120
  pinMode(MCP_EN_PIN, OUTPUT);
  digitalWrite(MCP_EN_PIN, HIGH);
  
  // Attente Device Reset Timer du MCP2120
  delay(50);
  
  // Configuration LEDs
  pinMode(PIN_SPEED, OUTPUT);
  pinMode(PIN_STOP, OUTPUT);
  
  digitalWrite(PIN_STOP, LOW);
  analogWrite(PIN_SPEED, 0);
  
  dernierSignalMillis = millis();
  
  Serial.println("=== RÉCEPTEUR avec MCP2120 ===");
  Serial.println("Baudrate: 9600");
  Serial.println("Commandes: S=Stop | G=Go");
}

void loop() {
  // Réception
  if (Serial2.available()) {
    vitesseRecue = Serial2.read();
    dernierSignalMillis = millis();
    
    Serial.print("REÇU : ");
    Serial.println(vitesseRecue);
  }

  // Commandes manuelles
  if (Serial.available()) {
    char c = toupper(Serial.read());
    if (c == 'S') obstacleDetecte = true;
    if (c == 'G') obstacleDetecte = false;
  }

  // Sécurité fail-safe
  bool perteSignal = (millis() - dernierSignalMillis > 1100);

  // Logique LEDs
  if (obstacleDetecte || perteSignal) {
    digitalWrite(PIN_STOP, HIGH);
    analogWrite(PIN_SPEED, 0);
    if (perteSignal) Serial.println("⚠ PERTE SIGNAL");
  } else {
    digitalWrite(PIN_STOP, LOW);
    analogWrite(PIN_SPEED, vitesseRecue);
  }
  
  delay(50);
}
