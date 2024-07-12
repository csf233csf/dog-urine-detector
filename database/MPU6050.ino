#include <Wire.h>
#include <MPU6050.h>
#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h>
#include <ESP8266HTTPClient.h>

const char* ssid = "YOUR_WIFI_SSID";  // 替换为您的WiFi SSID
const char* password = "YOUR_WIFI_PASSWORD";  // 替换为您的WiFi密码

const char* serverName = "http://www.kanohamnos.site/upload";  // 替换为您的服务器URL

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
    while (1);
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("Connected to Wi-Fi");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // 读取MPU6050数据
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // 准备要发送的数据
    String jsonData = "{\"acceleration\":{\"x\":" + String(ax) + ", \"y\":" + String(ay) + ", \"z\":" + String(az) + "},"
                      "\"gyroscope\":{\"x\":" + String(gx) + ", \"y\":" + String(gy) + ", \"z\":" + String(gz) + "}}";

    // 通过REST API调用
    WiFiClient client;
    HTTPClient http;
    Serial.println("URL: " + String(serverName));

    http.begin(client, serverName);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonData);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println(httpResponseCode);
      Serial.println(response);
    } else {
      Serial.print("Error on sending POST: ");
      Serial.println(httpResponseCode);
      Serial.println(http.errorToString(httpResponseCode).c_str());
    }

    http.end();
  } else {
    Serial.println("Error in WiFi connection");
  }

  delay(5000);  // 每10秒发送一次数据
}