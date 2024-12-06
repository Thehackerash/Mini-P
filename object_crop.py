import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")


output_dir = "cropped_objects"
filePath = "coordinatesLogFile.txt"
os.makedirs(output_dir, exist_ok=True)

<<<<<<< HEAD
def store_coordinates(file_path, coordinates, index):
    
 
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write("Index\tCoordinates (x1, y1, x2, y2)\n")

    
    with open(file_path, "a") as file:
        file.write(f"{index}\t({coordinates[0]}, {coordinates[1]}, {coordinates[2]}, {coordinates[3]})\n")

=======
>>>>>>> 1e66726608fd98d7d3c16d4a23200069a35295fd

def crop_and_save_objects(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    results = model(image)
    dict = {}  # store object_path:coordinates
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        dict[str(idx + 1)] = [x1, y1, x2, y2]
        cropped_object = image[y1:y2, x1:x2]
        object_path = os.path.join(output_dir, f"object_{idx + 1}.jpg")
        store_coordinates(filePath, (x1, y1, x2, y2), idx+1)
        cv2.imwrite(object_path, cropped_object)
        print(f"Saved: {object_path}")
    with open("coordinates.txt", "w") as f:
        f.write(str(dict))


<<<<<<< HEAD
input_image = "test.webp"

# crop_and_save_objects(input_image, output_dir)

=======
# input_image = "bottle.jpeg"
# crop_and_save_objects(input_image, output_dir)
>>>>>>> 1e66726608fd98d7d3c16d4a23200069a35295fd
