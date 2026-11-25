import os
import io
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_PATH = "selfie_segmenter.tflite"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    urllib.request.urlretrieve(url, MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Clothing Removal</title>
    <style>
        body { font-family: sans-serif; background: #0e0f12; color: #fff; padding: 20px; text-align: center; }
        form { background: #16181d; padding: 20px; border-radius: 10px; display: inline-block; }
        input[type=file] { margin: 10px 0; color: #fff; }
        button { background: #4ade80; color: #000; border: none; padding: 10px 20px; font-weight: bold; border-radius: 5px; cursor: pointer; }
        img { max-width: 100%; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <h2>Clothing Removal</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required><br>
        <button type="submit">Process Image</button>
    </form>
    {% if result_img %}
    <div>
        <h3>Output:</h3>
        <img src="data:image/jpeg;base64,{{ result_img }}" alt="Result">
    </div>
    {% endif %}
</body>
</html>
"""

def process_image(bgr):
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.ImageSegmenterOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = seg.segment(mp_img)
        mask = result.category_mask.numpy_view()
        person_mask = (mask > 0).astype(np.uint8)

    # Skin mask
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    skin = cv2.inRange(hsv, lower, upper)
    skin = cv2.GaussianBlur(skin, (5, 5), 0)
    skin = (skin > 127).astype(np.uint8)

    # Clothing = Person minus Skin
    clothing = cv2.bitwise_and(person_mask, cv2.bitwise_not(skin))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clothing = cv2.morphologyEx(clothing, cv2.MORPH_CLOSE, kernel, iterations=2)
    clothing = cv2.morphologyEx(clothing, cv2.MORPH_OPEN, kernel, iterations=1)

    # Inpaint
    inpainted = cv2.inpaint(bgr, clothing * 255, 5, cv2.INPAINT_TELEA)
    out = bgr.copy()
    out[clothing == 1] = inpainted[clothing == 1]
    return out

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file: return "No file", 400
        arr = np.frombuffer(file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if bgr is not None:
            out = process_image(bgr)
            _, enc = cv2.imencode(".jpg", out)
            b64 = base64.b64encode(enc).decode("utf-8")
            return render_template_string(HTML_PAGE, result_img=b64)
            
    return render_template_string(HTML_PAGE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))