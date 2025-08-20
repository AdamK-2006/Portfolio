from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://widefm.capsulecrm.com"}})

TEMPLATES_DIR = os.environ.get('TEMPLATE_DIR', 'templates')

@app.route('/flask/templates')
def list_templates():
    files = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith('.docx')]
    return jsonify(files)

@app.route('/flask/templates/<filename>')
def serve_template(filename):
    return send_from_directory(TEMPLATES_DIR, filename)


if __name__ == '__main__':
    app.run(port=8000)
