
import time
import csv
from pathlib import Path
from ultralytics import YOLO

# Incarcam modelele
models = {
    "PyTorch": YOLO("yolo26n.pt"),
    "ONNX": YOLO("yolo26n.onnx")
}

# Setul de testare
dataset_path = Path("datasets/benchmark_images")
images = sorted(dataset_path.glob("*.jpg"))

if not images:
    raise RuntimeError("Nu s-au gasit imagini in dataset!")

print(f"Numar imagini: {len(images)}")

# Setari
warmup_runs = 5
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

csv_path = results_folder / "benchmark_dataset_yolo26.csv"

benchmark_results = []

for model_name, model in models.items():

    print(f"\nBenchmark: {model_name}")

    # Warm-up
    for _ in range(warmup_runs):
        model.predict(str(images[0]), imgsz=640, verbose=False)

    total_times = []
    inference_times = []

    for index, image in enumerate(images, start=1):

        start = time.perf_counter()

        results = model.predict(
            str(image),
            imgsz=640,
            verbose=False
        )

        end = time.perf_counter()

        total = (end - start) * 1000
        inference = results[0].speed["inference"]
        fps = 1000 / total
        detections = len(results[0].boxes)

        total_times.append(total)
        inference_times.append(inference)

        benchmark_results.append([
            model_name,
            image.name,
            total,
            inference,
            fps,
            detections
        ])

        if index % 50 == 0:
            print(f"Procesate: {index}/{len(images)}")

    avg_total = sum(total_times) / len(total_times)
    avg_inference = sum(inference_times) / len(inference_times)
    avg_fps = 1000 / avg_total

    print(f"\n{model_name} - REZULTATE")
    print(f"Total latency: {avg_total:.2f} ms")
    print(f"Inference latency: {avg_inference:.2f} ms")
    print(f"FPS: {avg_fps:.2f}")

# Salvam rezultatele in CSV
with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Image",
        "Total latency (ms)",
        "Inference latency (ms)",
        "FPS",
        "Detections"
    ])

    writer.writerows(benchmark_results)

print(f"\nRezultatele au fost salvate in: {csv_path}")
