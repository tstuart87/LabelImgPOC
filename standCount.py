from ultralytics import YOLO
import cv2
import os

# Terminal command to train YOLO model
# yolo detect train data=corn.yaml model=yolov8n.pt epochs=50 imgsz=640
# yolo detect train data=corn.yaml model=yolov8n.pt epochs=50 imgsz=1536   - might need to try this one
# imgsz sets the size of the image that the model trains with
# Delete train.cache and val.cache if you change the training images and annotations. YOLO will auto-regenerate them.

# images/train - train YOLO model. Need 16
# images/val - these images will be processed(counted) during training - not in Prod. Need 4
# labels/train - labelimg text files created - correspond to the images in images/train
# labels/val - labelimg text files created - correspond to the images in images/val

# Load the trained model
# MODELS TO SAVE:
# train5 - trained on row, single, double - getting a few rows but no plants
# train6 - trained on row, single, double - getting a few rows but no plants - more images trained than train5
# train13 - trained on corn - not row, single, double - pretty close on total count if you change confidence
model = YOLO("runs/detect/train/weights/best.pt")

# List of images to process for training and validation - comment this out once training and validation are finished, and we want to process stand counts on a directory of production images
# image_paths = [
#     r"C:\Users\stuar\PycharmProjects\LabelImgPOC\dataset\images\val\standCountImageD09_val.png",
#     r"C:\Users\stuar\PycharmProjects\LabelImgPOC\dataset\images\val\standCountImageL10_val.png"
# ]

# Prod image_paths - below is where images need to be stored for Prod counting after model is trained.
prod_folder = r"C:\Users\stuar\PycharmProjects\LabelImgPOC\ProdImages"
image_paths = [os.path.join(prod_folder, f) for f in os.listdir(prod_folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))]

# Class indices based for training:
# 0 = row
# 1 = single plant
# 2 = double plant
# ROW = 0
# SINGLE = 1
# DOUBLE = 2
CORN = 0

for img_path in image_paths:

    # Make sure image exists
    if not os.path.exists(img_path):
        print(f"ERROR: File not found → {img_path}")
        continue

    print(f"\nProcessing image: {img_path}")

    # Run inference
    # results = model(img_path, imgsz=1536, conf=0.2)[0]
    # results = model(img_path, conf=0.013445)[0] # fairly accurate with non-enhanced images
    results = model(img_path, conf=0.009755)[0] # extremely accurate with dark enhanced images, not good for light images

    num_rows = 0
    num_plants = 0

    # Loop through detections
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == CORN:
            num_plants += 1

    # Print results for this image
    # print("  Rows detected:", num_rows)
    print("  Plants detected:", num_plants)

    # Show annotated image
    annotated = results.plot()
    cv2.imshow(f"Predictions - {img_path}", annotated)

cv2.destroyAllWindows()
