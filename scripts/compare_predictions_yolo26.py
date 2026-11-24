
from pathlib import Path
from ultralytics import YOLO
import csv

pytorch_model = YOLO("yolo26n.pt")
onnx_model = YOLO("yolo26n.onnx")

images = sorted(Path("datasets/benchmark_images").glob("*.jpg"))

output_file = "results/prediction_comparison.csv"

rows = []

for index, image in enumerate(images, start=1):

    pytorch_results = pytorch_model.predict(
        str(image), imgsz=640, verbose=False
    )[0]

    onnx_results = onnx_model.predict(
        str(image), imgsz=640, verbose=False
    )[0]

    pytorch_boxes = pytorch_results.boxes
    onnx_boxes = onnx_results.boxes

    pytorch_classes = pytorch_boxes.cls.cpu().tolist()
    onnx_classes = onnx_boxes.cls.cpu().tolist()

    pytorch_conf = pytorch_boxes.conf.cpu().tolist()
    onnx_conf = onnx_boxes.conf.cpu().tolist()

    pytorch_counts = {}
    onnx_counts = {}

    for cls in pytorch_classes:
        name = pytorch_results.names[int(cls)]
        pytorch_counts[name] = pytorch_counts.get(name, 0) + 1

    for cls in onnx_classes:
        name = onnx_results.names[int(cls)]
        onnx_counts[name] = onnx_counts.get(name, 0) + 1

    same_classes = pytorch_counts == onnx_counts

    rows.append([
        image.name,
        len(pytorch_boxes),
        len(onnx_boxes),
        same_classes,
        sum(pytorch_conf) / len(pytorch_conf) if pytorch_conf else 0,
        sum(onnx_conf) / len(onnx_conf) if onnx_conf else 0
    ])

    if index % 50 == 0:
        print(f"Comparate: {index}/{len(images)}")

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "PyTorch detections",
        "ONNX detections",
        "Same classes and counts",
        "PyTorch average confidence",
        "ONNX average confidence"
    ])

    writer.writerows(rows)

same_count = sum(1 for row in rows if row[3])

print("\nCOMPARATIE FINALA")
print(f"Total imagini: {len(rows)}")
print(f"Imagini cu aceleasi clase si numere: {same_count}")
print(f"Procent: {same_count / len(rows) * 100:.2f}%")
print(f"CSV salvat: {output_file}")
