#include <Arduino.h>

// Connexions ESP32 vers MCP2120
#define MCP_TX_PIN  17  
#define MCP_RX_PIN  16  
#define MCP_EN_PIN  33  

// OLED definitions
#include <Wire.h>             
#include "SSD1306Wire.h"       

SSD1306Wire display(0x3c, 5, 4);  

void setup() {
  Serial.begin(9600);   // Debug
  Serial2.begin(9600, SERIAL_8N1, MCP_RX_PIN, MCP_TX_PIN);
  
  // Enable du MCP2120
  pinMode(MCP_EN_PIN, OUTPUT);
  digitalWrite(MCP_EN_PIN, HIGH);
  
  // OLED init
  display.init();
  display.flipScreenVertically();
  display.setFont(ArialMT_Plain_16);
  
  // Écran d'accueil
  display.clear();
  display.setTextAlignment(TEXT_ALIGN_CENTER);
  display.drawString(64, 0, "EMETTEUR IrDA");
  display.setFont(ArialMT_Plain_10);
  display.drawString(64, 25, "MCP2120");
  display.drawString(64, 40, "Baudrate: 9600");
  display.display();
  
  delay(2000);
  
  Serial.println("=== ÉMETTEUR avec MCP2120 ===");
  Serial.println("Baudrate: 9600");
}

void loop() {
  uint8_t vitesses[] = {0, 51, 102, 153, 204, 255};
  
  for (int i = 0; i < 6; i++) {
    // Affichage console
    Serial.print("Envoi : ");
    Serial.println(vitesses[i]);
    
    // Envoi IrDA
    Serial2.write(vitesses[i]);
    Serial2.flush();
    
    // Affichage OLED
    display.clear();
    
    // Titre
    display.setFont(ArialMT_Plain_10);
    display.setTextAlignment(TEXT_ALIGN_CENTER);
    display.drawString(64, 0, "EMISSION IrDA");
    
    // Ligne de séparation
    display.drawHorizontalLine(0, 12, 128);
    
    // Vitesse (grande police)
    display.setFont(ArialMT_Plain_24);
    display.setTextAlignment(TEXT_ALIGN_CENTER);
    display.drawString(64, 20, String(vitesses[i]));
    
    // Label "Vitesse envoyée"
    display.setFont(ArialMT_Plain_10);
    display.drawString(64, 48, "Vitesse envoyee");
    
    display.display();
    
    delay(1000);
  }
  
  // Fin de cycle
  display.clear();
  display.setFont(ArialMT_Plain_16);
  display.setTextAlignment(TEXT_ALIGN_CENTER);
  display.drawString(64, 20, "Cycle termine");
  display.display();
  
  Serial.println("--- Cycle terminé ---");
  delay(50);
}