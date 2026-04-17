How to Run

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/iot-predictive-maintenance-api.git
cd iot-predictive-maintenance-api
2. Install Dependencies
pip install -r requirements.txt
3. Retrain Model (IMPORTANT)
python retrain.py
4. Run Flask App
python app.py
Open in browser:
http://127.0.0.1:5001
📡 API Usage (Postman)
Endpoint
POST /predict
Sample Input
{
  "vibration": 5.5,
  "temperature": 72,
  "current": 16,
  "acoustic": 78,
  "IMF_1": 1.05,
  "IMF_3": 1.18,
  "timestamp": 92,
  "IMF_2": 1.10,
  "pressure": 114,
  "humidity": 63
}
Output
{
  "status": "success",
  "prediction": 1,
  "prediction_label": "Fault Detected"
}
🐳 Docker Setup
Build Image
docker build -t iot-app .
Run Container
docker run -p 5001:5001 iot-app
📊 Features Used
vibration
temperature
current
acoustic
IMF_1
IMF_2
IMF_3
timestamp
pressure (new)
humidity (new)
