
import csv
import time
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
IMAGES_DIR = Path("datasets/test_images")
RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "yolo26_benchmark_results.csv"

IMAGE_SIZE = 640
WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

model = YOLO(MODEL_PATH)

image_paths = sorted(IMAGES_DIR.glob("*.jpg"))

if not image_paths:
    raise FileNotFoundError("Nu exista imagini de test.")

RESULTS_DIR.mkdir(exist_ok=True)

print("Model: YOLO26n")
print("Dispozitiv: CPU")
print("Precizie: FP32")
print(f"Numar imagini: {len(image_paths)}")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "model",
        "device",
        "precision",
        "image",
        "image_size",
        "runs",
        "avg_total_latency_ms",
        "avg_inference_latency_ms",
        "fps"
    ])

    for image_path in image_paths:
        for _ in range(WARMUP_RUNS):
            model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )

        total_times = []
        inference_times = []

        for _ in range(BENCHMARK_RUNS):
            start = time.perf_counter()

            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False
            )

            end = time.perf_counter()

            total_times.append((end - start) * 1000)
            inference_times.append(result[0].speed["inference"])

        avg_total = sum(total_times) / len(total_times)
        avg_inference = sum(inference_times) / len(inference_times)
        fps = 1000 / avg_total

        writer.writerow([
            "YOLO26n",
            "CPU",
            "FP32",
            image_path.name,
            IMAGE_SIZE,
            BENCHMARK_RUNS,
            round(avg_total, 2),
            round(avg_inference, 2),
            round(fps, 2)
        ])

        print(f"\nImagine: {image_path.name}")
        print(f"Latență totală: {avg_total:.2f} ms")
        print(f"Latență inferență: {avg_inference:.2f} ms")
        print(f"FPS: {fps:.2f}")

print("\nBenchmark terminat!")
print(f"Rezultatele sunt salvate in {CSV_PATH}")
