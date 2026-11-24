
import argparse
import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO


IMG_SIZE = 640
WARMUP_RUNS = 10
REPEATS = 3

IMAGE_DIR = Path("datasets/benchmark_images")
RESULTS_DIR = Path("results")

MODELS = [
    ("PyTorch FP32", "yolo26n.pt"),
    ("TensorRT FP32", "yolo26n_fp32.engine"),
    ("TensorRT FP16", "yolo26n_fp16.engine"),
]


def synchronize():
    torch.cuda.synchronize()


def get_images():
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError("Nu exista imagini in benchmark_images.")

    return images


def run_prediction(model, image_path):
    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        device=0,
        conf=0.25,
        iou=0.7,
        verbose=False,
        save=False
    )

    return results[0]


def benchmark_model(model_name, model_path, images):
    print(f"\n{'=' * 50}")
    print(f"Model: {model_name}")
    print(f"Fisier: {model_path}")
    print(f"{'=' * 50}")

    model = YOLO(model_path, task="detect")

    # Warmup
    print(f"Warmup: {WARMUP_RUNS} executii...")

    for _ in range(WARMUP_RUNS):
        run_prediction(model, images[0])
        synchronize()

    total_times = []
    inference_times = []
    preprocess_times = []
    postprocess_times = []
    detection_counts = []

    # Benchmark
    total_runs = len(images) * REPEATS
    current_run = 0

    print(f"Benchmark: {len(images)} imagini x {REPEATS} repetari")

    for image_path in images:
        for _ in range(REPEATS):
            synchronize()
            start = time.perf_counter()

            result = run_prediction(model, image_path)

            synchronize()
            end = time.perf_counter()

            total_ms = (end - start) * 1000

            total_times.append(total_ms)

            speed = result.speed

            preprocess_times.append(speed["preprocess"])
            inference_times.append(speed["inference"])
            postprocess_times.append(speed["postprocess"])

            detection_counts.append(len(result.boxes))

            current_run += 1

        if current_run % 150 == 0 or current_run == total_runs:
            print(f"Progres: {current_run}/{total_runs}")

    avg_total = sum(total_times) / len(total_times)
    avg_inference = sum(inference_times) / len(inference_times)
    avg_preprocess = sum(preprocess_times) / len(preprocess_times)
    avg_postprocess = sum(postprocess_times) / len(postprocess_times)
    avg_detections = sum(detection_counts) / len(detection_counts)

    fps_total = 1000 / avg_total
    fps_inference = 1000 / avg_inference

    print(f"\nRezultate {model_name}")
    print(f"Total latency: {avg_total:.2f} ms")
    print(f"Inference latency: {avg_inference:.2f} ms")
    print(f"Preprocessing: {avg_preprocess:.2f} ms")
    print(f"Postprocessing: {avg_postprocess:.2f} ms")
    print(f"FPS total: {fps_total:.2f}")
    print(f"FPS inference: {fps_inference:.2f}")
    print(f"Average detections: {avg_detections:.2f}")

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_file": model_path,
        "images": len(images),
        "repeats": REPEATS,
        "warmup": WARMUP_RUNS,
        "total_latency_ms": round(avg_total, 3),
        "inference_latency_ms": round(avg_inference, 3),
        "preprocess_ms": round(avg_preprocess, 3),
        "postprocess_ms": round(avg_postprocess, 3),
        "fps_total": round(fps_total, 3),
        "fps_inference": round(fps_inference, 3),
        "average_detections": round(avg_detections, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["pt", "fp32", "fp16"],
        default=["pt", "fp32", "fp16"]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nu este disponibil. Acest benchmark trebuie rulat pe Jetson."
        )

    print("CUDA disponibil:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    images = get_images()
    print("Numar imagini:", len(images))

    RESULTS_DIR.mkdir(exist_ok=True)

    selected_models = {
        "pt": MODELS[0],
        "fp32": MODELS[1],
        "fp16": MODELS[2],
    }

    results = []

    for key in args.models:
        model_name, model_path = selected_models[key]

        if not Path(model_path).exists():
            print(f"\nLipseste {model_path}. Sar peste acest model.")
            continue

        result = benchmark_model(model_name, model_path, images)
        results.append(result)

    if not results:
        raise RuntimeError("Nu a fost testat niciun model.")

    output_file = RESULTS_DIR / "benchmark_jetson_yolo26.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nBenchmark finalizat.")
    print("Rezultate salvate in:", output_file)


if __name__ == "__main__":
    main()
