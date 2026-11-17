
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

print("Se exporta YOLO26n in format ONNX...")

model.export(
    format="onnx",
    imgsz=640,
    dynamic=False
)

print("Export ONNX finalizat!")
