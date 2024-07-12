#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

const char* ssid = "YOUR_WIFI_SSID";  // 替换为您的WiFi SSID
const char* password = "YOUR_WIFI_PASSWORD";  // 替换为您的WiFi密码
const char* serverName = "http://www.kanohamnos.site/upload_water";  // 替换为您的服务器URL

const int sensorPin = A0;  // 水位传感器连接到 A0 引脚

void setup() {
  Serial.begin(115200);  // 初始化串口通讯

  // 开始连接Wi-Fi
  Serial.println();
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int attempts = 0;  // 记录尝试次数
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {  // 尝试20次
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("");
    Serial.println("Failed to connect to WiFi");
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // 读取水位传感器数据
    int sensorValue = analogRead(sensorPin);
    float voltage = sensorValue * (5.0 / 1023.0);  // 将模拟值转换为电压值

    // 打印调试信息
    Serial.print("Water Level (analog value): ");
    Serial.print(sensorValue);
    Serial.print("   Voltage: ");
    Serial.println(voltage);

    // 将数据转换为JSON字符串
    String jsonData = "{\"water_level\":" + String(sensorValue) + "}";

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

  delay(10000);  // 每10秒发送一次数据
}
