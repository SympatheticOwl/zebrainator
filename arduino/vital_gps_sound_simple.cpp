#include "Adafruit_GPS.h"
#include "PulseOximeter.h"

#define PULSE_OXI_PIN 2
#define SOUND_PIN 3
Adafruit_GPS GPS;
PulseOximeter pox;
int sound_val = 0;

void setup() {
    Serial.begin(9600);
    pinMode(PULSE_OXI_PIN, INPUT);
    pinMode(SOUND_PIN, INPUT);
    if (!pox.begin()) {
        Serial.println("Vitals sensor initialization failed");
    } else {
        Serial.println("Vitals sensor initialized");
    }
    Serial.println("Zebra monitoring system initialized");
}

void loop() {
    if (Serial.available() > 0) {
        string msg = Serial.readMessage();
        Serial.println("Received message: " + msg);

        // sound
        sound_val = digitalRead(SOUND_PIN);
        Serial.println("Sound: " + to_string(sound_val));
        delay(1000);

        // GPS
        string gpsResult = GPS.read();
        Serial.println("GPS: " + gpsResult);

         // vitals
        if (pox.begin()) {
            string spO2_res = pox.getSpO2();
            string hb_res = pox.getHeartRate();
            Serial.println("Heart Rate: " + hb_res);
            Serial.println("SpO2: " + spO2_res);
        } else {
            Serial.println("No Oxygen data");
        }
    } else {
        Serial.println("Serial Data Unavailable");
    }
    Serial.println("---");
    //TODO: should poll data for 2-3 minutes every hour NOT every 2 minutes
    delay(120000);wwd
}