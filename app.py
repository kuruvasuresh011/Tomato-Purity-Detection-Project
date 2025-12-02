from flask import Flask, render_template, request
import os
import uuid
from ultralytics import YOLO
import cv2

# Flask app init
app = Flask(__name__)

# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load your trained model
model = YOLO("best.pt")    # <-- make sure best.pt is in same folder

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No file selected"

    # Save uploaded image with unique name
    img_id = uuid.uuid4().hex
    input_path = os.path.join("static", f"input_{img_id}.jpg")
    file.save(input_path)

    # Run YOLO prediction
    results = model(input_path)

    # Annotate + save output
    output_path = os.path.join("static", f"result_{img_id}.jpg")
    annotated = results[0].plot()  # draw boxes
    cv2.imwrite(output_path, annotated)

    # Get detections list
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        detections.append({
            "class": name,
            "confidence": round(conf, 2)
        })

    return render_template(
        "index.html",
        result_img=f"result_{img_id}.jpg",
        detections=detections
    )


if __name__ == "__main__":
    app.run(debug=True)
