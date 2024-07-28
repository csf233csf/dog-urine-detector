from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sensor_data.db'
app.config['SQLALCHEMY_BINDS'] = {
    'water_sensor': 'sqlite:///water_sensor.db'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sensor_type = db.Column(db.String(50), nullable=False)
    value = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<SensorData {self.sensor_type} {self.value}>'

class WaterSensorData(db.Model):
    __bind_key__ = 'water_sensor'
    id = db.Column(db.Integer, primary_key=True)
    water_level = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<WaterSensorData {self.water_level}>'

with app.app_context():
    db.create_all()

@app.route('/upload', methods=['POST'])
def upload_data():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    print("Received data:", data)

    try:
        acceleration = data['acceleration']
        gyroscope = data['gyroscope']

        new_data_ax = SensorData(sensor_type='acceleration_x', value=acceleration['x'])
        new_data_ay = SensorData(sensor_type='acceleration_y', value=acceleration['y'])
        new_data_az = SensorData(sensor_type='acceleration_z', value=acceleration['z'])
        new_data_gx = SensorData(sensor_type='gyroscope_x', value=gyroscope['x'])
        new_data_gy = SensorData(sensor_type='gyroscope_y', value=gyroscope['y'])
        new_data_gz = SensorData(sensor_type='gyroscope_z', value=gyroscope['z'])

        db.session.add(new_data_ax)
        db.session.add(new_data_ay)
        db.session.add(new_data_az)
        db.session.add(new_data_gx)
        db.session.add(new_data_gy)
        db.session.add(new_data_gz)
        db.session.commit()

        return jsonify({"message": "Data uploaded successfully"}), 201
    except KeyError as e:
        return jsonify({"error": f"Missing key {str(e)}"}), 400

@app.route('/upload_water', methods=['POST'])
def upload_water_data():
    data = request.json
    if not data or 'water_level' not in data:
        return jsonify({"error": "Invalid data"}), 400

    print("Received water data:", data)

    try:
        water_level = data['water_level']

        new_water_data = WaterSensorData(water_level=water_level)
        db.session.add(new_water_data)
        db.session.commit()

        return jsonify({"message": "Water data uploaded successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
