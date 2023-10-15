import cv2
import sys
from ultralytics import YOLO
from cube import Cube, cubes_from_results

source = "Test_images/005.jpg"

image = cv2.imread(source)

model = YOLO("model/best.pt")
results = model.predict(source=source)

detections: [Cube] = cubes_from_results(results)

for result in results:

    result_image = result.plot()

    for box in result.boxes: 
        coordinates = box.xyxy.cpu().numpy()
        coordinates = coordinates[0]

        center_x = (coordinates[0] + coordinates[2])/2
        center_y = (coordinates[1] + coordinates[3])/2
        cv2.circle(result_image, (int(center_x), int(center_y)), 10, (255, 0, 0, 255), 2)

cv2.imshow('results', result_image)
cv2.waitKey(0)
sys.exit()