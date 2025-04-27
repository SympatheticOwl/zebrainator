#include "Adafruit_GPS.h"
#include "PulseOximeter.h"
#include "MQ135.h"
#include "DHT.h"

#define PULSE_OXI_PIN 2
#define HUMIDITY_PIN 3
#define AIR_QUALITY_DIGITAL_PIN 4

#define TEMP_SENSOR_PIN A1
#define AIR_QUALITY_PIN A3

Adafruit_GPS GPS;
PulseOximeter pox;
DHT dht(HUMIDITY_PIN);
MQ135 mq135(AIR_QUALITY_PIN);
int sound_val = 0;

void setup() {
    Serial.begin(9600);
    pinMode(0, OUTPUT);
    pinMode(PULSE_OXI_PIN, INPUT);
    pinMode(HUMIDITY_PIN, INPUT);
    pinMode(TEMP_SENSOR_PIN, INPUT);

    if (!pox.begin()) {
        Serial.println("Vitals sensor initialization failed");
    } else {
        Serial.println("Vitals sensor initialized");
    }

    dht.begin();

    Serial.println("Zebra monitoring system initialized");
}

void loop() {
    // temp sensor
    float temp = analogRead(TEMP_SENSOR_PIN) / 1024.0 * 5.0 * 100.0;
    string temp_message = "Temperature: " + to_string(temp);
    Serial.println(temp_message);
    Serial.sendMessage(0, temp_message);
    delay(1000);

    // air quality sensor
    float airQuality = mq135.getPPM();
    string air_qual_msg = "Air Quality (PPM): " + to_string(airQuality);
    Serial.println(air_qual_msg);
    Serial.sendMessage(0, air_qual_msg);
    delay(1000);

    // humidity sensor
    //bool isInCelcius = true;
    float h = dht.readHumidity() * 100; // Convert to %
    string hum_msg = "Humidity: " + to_string(h) + "%";
    Serial.println(hum_msg);
    Serial.sendMessage(0, hum_msg);
    delay(1000);

    if (Serial.available() > 0) {
        string msg = Serial.readMessage();
        Serial.println("Received message: " + msg);

        // GPS
        string gpsResult = GPS.read();
        string gps_msg = "GPS: " + gpsResult;
        Serial.println(gps_msg);
        Serial.sendMessage(0, gps_msg);
        delay(1000);

         // vitals
        if (pox.begin()) {
            string spO2_res = pox.getSpO2();
            string hb_res = pox.getHeartRate();
            string heart_rate = "Heart Rate: " + hb_res;
            Serial.println(heart_rate);
            string sp02 = "SpO2: " + spO2_res;
            Serial.println(sp02);
            Serial.sendMessage(0, heart_rate + "," + sp02);
        } else {
            string no_data_msg = "No Oxygen data";
            Serial.println(no_data_msg);
            Serial.sendMessage(0, no_data_msg);
        }
    } else {
        string no_serial_data_msg = "Serial Data Unavailable";
        Serial.println(no_serial_data_msg);
        Serial.sendMessage(0, no_serial_data_msg);
    }
    Serial.println("---");
    //TODO: should poll data for 2-3 minutes every hour NOT every 2 minutes
    delay(120000);
}
