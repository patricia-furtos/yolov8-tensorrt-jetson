
from ultralytics import YOLO
from pathlib import Path
import shutil

# Incarc modelul YOLOv8n
model = YOLO("yolov8n.pt")

# Rulez detectia pe o imagine de test
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device="cpu",
    save=True
)

# Copiez imaginea rezultata in folderul proiectului
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

saved_image = Path(results[0].save_dir) / "bus.jpg"
if saved_image.exists():
    shutil.copy(saved_image, output_dir / "first_detection.jpg")
    print("Imagine salvata in results/first_detection.jpg")
else:
    print("Detectia s-a executat. Verifica folderul runs/detect.")
