import requests
import time

# Hugging Face API base URL
HF_API = "https://huggingface.co/api/models/"

# Weights for NetScore calculation (must sum to ~1.0)
WEIGHTS = {
    "ramp_up_time": 0.15,
    "bus_factor": 0.15,
    "license": 0.15,
    "performance_claims": 0.15,
    "dataset_and_code_score": 0.1,
    "dataset_quality": 0.1,
    "code_quality": 0.1,
    "size_score": 0.1,
}


def compute_net_score(metrics: dict) -> float:
    """Compute weighted average of metrics (ignores latencies)."""
    size_avg = sum(metrics["size_score"].values()) / 4  # avg of 4 hardware targets
    score = 0
    score += WEIGHTS["ramp_up_time"] * metrics["ramp_up_time"]
    score += WEIGHTS["bus_factor"] * metrics["bus_factor"]
    score += WEIGHTS["license"] * metrics["license"]
    score += WEIGHTS["performance_claims"] * metrics["performance_claims"]
    score += WEIGHTS["dataset_and_code_score"] * metrics["dataset_and_code_score"]
    score += WEIGHTS["dataset_quality"] * metrics["dataset_quality"]
    score += WEIGHTS["code_quality"] * metrics["code_quality"]
    score += WEIGHTS["size_score"] * size_avg
    return round(score, 3)  # round for cleaner output


def compute_metrics_for_model(url: str) -> dict:
    """
    Given a Hugging Face model URL, fetch metadata and compute metrics.
    """
    model_name = url.split("/")[-1]
    start = time.time()

    try:
        response = requests.get(f"{HF_API}{model_name}", timeout=10)
        data = response.json()
    except Exception:
        data = {}

    # Example metric: ramp_up_time = proxy for README length
    ramp_up_time = len(data.get("cardData", {}).get("long_description", "")) / 1000
    ramp_up_time = min(ramp_up_time, 1.0)

    # Example metric: bus_factor = proxy for number of files in repo
    bus_factor = len(data.get("siblings", [])) / 10
    bus_factor = min(bus_factor, 1.0)

    # License check: 1 if exists, else 0
    license = 1.0 if data.get("license") else 0.0

    # Placeholders (to refine later)
    performance_claims = 0.5
    dataset_and_code_score = 0.5
    dataset_quality = 0.5
    code_quality = 0.5

    # Size score placeholder
    size_score = {
        "raspberry_pi": 0.5,
        "jetson_nano": 0.5,
        "desktop_pc": 0.5,
        "aws_server": 0.5,
    }

    # Compute NetScore using weights
    net_score = compute_net_score({
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "license": license,
        "performance_claims": performance_claims,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score
    })

    elapsed = int((time.time() - start) * 1000)

    result = {
        "name": model_name,
        "category": "MODEL",
        "net_score": net_score,
        "ramp_up_time": ramp_up_time,
        "bus_factor": bus_factor,
        "performance_claims": performance_claims,
        "license": license,
        "dataset_and_code_score": dataset_and_code_score,
        "dataset_quality": dataset_quality,
        "code_quality": code_quality,
        "size_score": size_score,
    }

    # Add latency metrics
    for key in list(result.keys()):
        if key not in ("name", "category", "size_score"):
            result[f"{key}_latency"] = elapsed

    return result
