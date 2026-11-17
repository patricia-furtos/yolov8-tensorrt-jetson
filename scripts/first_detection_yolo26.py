
from ultralytics import YOLO

# Incarcam modelul YOLO26 nano
model = YOLO("yolo26n.pt")

print("Model YOLO26n incarcat.")

# Testam detectia pe imaginile noastre
results = model.predict(
    source="datasets/test_images",
    imgsz=640,
    device="cpu",
    save=True,
    project="results",
    name="yolo26_first_detection",
    exist_ok=True
)

print("Detectiile YOLO26 au fost executate cu succes!")
print(f"Numar imagini procesate: {len(results)}")
