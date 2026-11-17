
from ultralytics import YOLO

# Incarcam modelul exportat in format ONNX
model = YOLO("yolo26n.onnx")

print("Modelul ONNX a fost incarcat.")

# Executam detectia pe imaginile de test
results = model.predict(
    source="datasets/test_images",
    imgsz=640,
    device="cpu",
    save=True,
    project="results",
    name="onnx_test",
    exist_ok=True
)

print("Testul ONNX s-a terminat!")
print(f"Numar imagini procesate: {len(results)}")
