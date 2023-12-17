from flask import Flask, render_template, request
import cv2
import numpy as np
from io import BytesIO
import base64
from scipy.ndimage import gaussian_filter, median_filter
from skimage import restoration

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    # Get the uploaded image file
    uploaded_file = request.files['image']
    
    # Read the image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Apply the selected filter
    selected_filter = request.form.get('filter')

    if selected_filter == 'canny':
        filtered_img = apply_canny_edge_detection(img)
    elif selected_filter == 'weiner':
        filtered_img = apply_weiner_filter(img)
    elif selected_filter == 'inverse':
        filtered_img = apply_inverse_filter(img)
    elif selected_filter == 'median':
        filtered_img = apply_median_filter(img)
    elif selected_filter == 'gaussian':
        filtered_img = apply_gaussian_filter(img)
    elif selected_filter == 'mean':
        filtered_img = apply_mean_filter(img)
    elif selected_filter == 'non_local_means':
        filtered_img = apply_non_local_means_filter(img)
    elif selected_filter == 'laplace':
        filtered_img = apply_laplace_filter(img)
    else:
        # Handle unknown filter
        return redirect(url_for('index'))

    # Convert the filtered image to base64 for displaying in HTML
    _, img_encoded = cv2.imencode('.png', filtered_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('result.html', img_base64=img_base64)

def apply_canny_edge_detection(img):
    return cv2.Canny(img, 100, 200)

def apply_weiner_filter(img):
    # Implement Weiner Filter here (you may need to adjust parameters)
    return restoration.wiener(img)

def apply_inverse_filter(img):
    # Implement Inverse Filter here (you may need to adjust parameters)
    # Example: return cv2.bitwise_not(img)
    return img

def apply_median_filter(img):
    return median_filter(img, size=3)

def apply_gaussian_filter(img):
    return gaussian_filter(img, sigma=1)

def apply_mean_filter(img):
    return cv2.blur(img, (3, 3))

def apply_non_local_means_filter(img):
    # Implement Non-local Means Filter here (you may need to adjust parameters)
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def apply_laplace_filter(img):
    return cv2.Laplacian(img, cv2.CV_64F)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
