
from pathlib import Path
from ultralytics import YOLO
import csv
import numpy as np

pytorch_model = YOLO("yolo26n.pt")
onnx_model = YOLO("yolo26n.onnx")

images = sorted(Path("datasets/benchmark_images").glob("*.jpg"))

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


rows = []
total_matches = 0
total_boxes = 0
all_ious = []

for index, image in enumerate(images, start=1):

    pt = pytorch_model.predict(str(image), imgsz=640, verbose=False)[0]
    onnx = onnx_model.predict(str(image), imgsz=640, verbose=False)[0]

    pt_boxes = pt.boxes.xyxy.cpu().numpy()
    pt_classes = pt.boxes.cls.cpu().numpy()

    onnx_boxes = onnx.boxes.xyxy.cpu().numpy()
    onnx_classes = onnx.boxes.cls.cpu().numpy()

    used_onnx = set()
    matches = []

    for i, pt_box in enumerate(pt_boxes):
        best_iou = 0
        best_index = -1

        for j, onnx_box in enumerate(onnx_boxes):
            if j in used_onnx:
                continue

            if pt_classes[i] != onnx_classes[j]:
                continue

            iou = calculate_iou(pt_box, onnx_box)

            if iou > best_iou:
                best_iou = iou
                best_index = j

        if best_iou >= 0.5:
            used_onnx.add(best_index)
            matches.append(best_iou)

    matched = len(matches)
    denominator = max(len(pt_boxes), len(onnx_boxes))

    agreement = matched / denominator * 100 if denominator else 100

    total_matches += matched
    total_boxes += denominator
    all_ious.extend(matches)

    rows.append([
        image.name,
        len(pt_boxes),
        len(onnx_boxes),
        matched,
        round(agreement, 2),
        round(np.mean(matches), 4) if matches else 0
    ])

    if index % 50 == 0:
        print(f"Comparate: {index}/{len(images)}")

with open("results/box_comparison.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Image",
        "PyTorch boxes",
        "ONNX boxes",
        "Matched boxes",
        "Agreement (%)",
        "Average IoU"
    ])
    writer.writerows(rows)

print("\nREZULTATE")
print(f"Matched boxes: {total_matches}")
print(f"Total boxes: {total_boxes}")
print(f"Agreement: {total_matches / total_boxes * 100:.2f}%")
print(f"Average IoU: {np.mean(all_ious):.4f}" if all_ious else "No matches")
print("CSV: results/box_comparison.csv")
