
import time
from ultralytics import YOLO

# Incarcam modelele
pytorch_model = YOLO("yolo26n.pt")
onnx_model = YOLO("yolo26n.onnx")

# Imagini de test
images = [
    "datasets/test_images/bus.jpg",
    "datasets/test_images/zidane.jpg"
]

# Setari benchmark
warmup_runs = 5
benchmark_runs = 30


def benchmark(model, image, model_name):
    total_times = []
    inference_times = []

    # Warm-up: executii care nu intra in calculul mediei
    for _ in range(warmup_runs):
        model.predict(image, imgsz=640, verbose=False)

    # Masuratori
    for _ in range(benchmark_runs):
        start = time.perf_counter()

        results = model.predict(image, imgsz=640, verbose=False)

        end = time.perf_counter()

        total_times.append((end - start) * 1000)
        inference_times.append(results[0].speed["inference"])

    # Calculam mediile
    avg_total = sum(total_times) / len(total_times)
    avg_inference = sum(inference_times) / len(inference_times)

    # Calculam FPS
    fps = 1000 / avg_total

    # Numarul de detectii de la ultima executie
    detection_count = len(results[0].boxes)

    print(f"\n{model_name}")
    print(f"Total latency: {avg_total:.2f} ms")
    print(f"Inference latency: {avg_inference:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Detections: {detection_count}")

    return avg_total, avg_inference, fps, detection_count


# Comparam cele doua modele pe fiecare imagine
for image in images:
    print(f"\n========== {image} ==========")

    benchmark(pytorch_model, image, "PyTorch")
    benchmark(onnx_model, image, "ONNX")
