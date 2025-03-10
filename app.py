import os
import io
import base64
import tempfile
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Import our DETR vision package
from src.detr_vision.model import DetrObjectDetector
from src.detr_vision.visualization import draw_detections

# Initialize Flask app
app = Flask(__name__)

# Initialize the object detector (will be loaded when first needed)
detector = None


def get_detector():
    """Lazy-load the model only when needed"""
    global detector
    if detector is None:
        # Use CPU by default for deployment (unless you configure GPU on Render)
        detector = DetrObjectDetector(device="cpu")
    return detector


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    """API endpoint for object detection"""
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    # If user submits an empty form
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Get detection threshold from form, default to 0.5
    threshold = float(request.form.get('threshold', 0.5))

    # Read and process the image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Perform detection
    detector = get_detector()
    detections = detector.detect(img, threshold=threshold)

    # Draw detections on the image
    result_img = draw_detections(img_cv, detections, confidence_threshold=0.0)

    # Convert back to base64 for display
    _, buffer = cv2.imencode('.jpg', result_img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    # Prepare detection results for JSON response
    results = []
    for label, score, box in zip(detections['labels'], detections['scores'], detections['boxes']):
        results.append({
            'label': label,
            'confidence': float(score),
            'box': box.tolist()
        })

    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'detections': results
    })


if __name__ == '__main__':
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)