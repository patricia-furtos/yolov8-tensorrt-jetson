
from ultralytics import YOLO
from pathlib import Path
import time
import statistics
import csv

# Cai catre fisiere
IMAGE_PATH = Path("datasets/test_images/bus.jpg")

# Configurare test
MODEL_NAME = "yolov8n.pt"
IMAGE_SIZE = 640
WARMUP_RUNS = 5
TEST_RUNS = 30

# Verificam daca imaginea exista
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Imaginea nu a fost gasita: {IMAGE_PATH}")

# Incarcam modelul
model = YOLO(MODEL_NAME)

print("Model incarcat.")
print("Dispozitiv: CPU")
print("Precizie: FP32")

# Warm-up: rulam modelul de cateva ori inainte de masuratori
for _ in range(WARMUP_RUNS):
    model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

print("Warm-up terminat.")

# Masuram timpul pentru fiecare rulare
total_times = []
inference_times = []

for i in range(TEST_RUNS):
    start = time.perf_counter()

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False
    )

    end = time.perf_counter()

    total_times.append((end - start) * 1000)
    inference_times.append(results[0].speed["inference"])

# Calculam mediile
avg_total = statistics.mean(total_times)
avg_inference = statistics.mean(inference_times)
fps = 1000 / avg_total

# Afisam rezultatele
print("\n===== PYTORCH FP32 BENCHMARK =====")
print(f"Numar teste: {TEST_RUNS}")
print(f"Dimensiune imagine: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Latenta totala medie: {avg_total:.2f} ms")
print(f"Latenta inferentei medie: {avg_inference:.2f} ms")
print(f"FPS estimat (pe baza latentei totale): {fps:.2f}")

# Salvam o imagine cu detectiile
model.predict(
    source=str(IMAGE_PATH),
    imgsz=IMAGE_SIZE,
    device="cpu",
    save=True,
    project="results",
    name="pytorch_fp32",
    exist_ok=True,
    verbose=False
)

print("\nImaginea cu detectii a fost salvata in results/pytorch_fp32/")
# Salvam rezultatele in CSV
csv_path = Path("results/benchmark_results.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, mode="a", newline="") as file:
    writer = csv.writer(file)

    if csv_path.stat().st_size == 0:
        writer.writerow([
            "model",
            "device",
            "precision",
            "image_size",
            "runs",
            "avg_total_latency_ms",
            "avg_inference_latency_ms",
            "fps"
        ])

    writer.writerow([
        MODEL_NAME,
        "CPU",
        "FP32",
        IMAGE_SIZE,
        TEST_RUNS,
        round(avg_total, 2),
        round(avg_inference, 2),
        round(fps, 2)
    ])

print(f"Rezultatele au fost salvate in {csv_path}")