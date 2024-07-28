from flask import Flask, request, jsonify
from threading import Timer

app = Flask(__name__)

urination_status = False

def reset_status():
    global urination_status
    urination_status = False
    print("Urination status reset to false")

@app.route('/urination_status', methods=['GET'])
def get_urination_status():
    return jsonify(str(urination_status).lower())

@app.route('/urination_status', methods=['POST'])
def set_urination_status():
    global urination_status
    if 'status' in request.json:
        urination_status = request.json['status']
        if urination_status:
            Timer(30.0, reset_status).start()
        return jsonify(success=True), 200
    else:
        return jsonify(success=False, message="Invalid request"), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
