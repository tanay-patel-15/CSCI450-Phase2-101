import pytest
from src.metrics import compute_metrics_for_model

# --- Basic Functionality ---
def test_metrics_returns_dict():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    assert isinstance(result, dict)

def test_metrics_has_required_fields():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    for key in ["name", "category", "net_score", "ramp_up_time", "license"]:
        assert key in result

def test_metrics_name_extraction():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    assert result["name"] == "gemma-3-270m"

# --- Multiple Inputs ---
@pytest.mark.parametrize("url", [
    "https://huggingface.co/google/model1",
    "https://huggingface.co/org/model2",
    "https://huggingface.co/anon/model3",
])
def test_metrics_multiple_models(url):
    result = compute_metrics_for_model(url)
    assert result["category"] == "MODEL"

# --- Latency Fields ---
def test_metrics_latency_fields_exist():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    latency_keys = [k for k in result.keys() if k.endswith("_latency")]
    assert len(latency_keys) > 0

def test_metrics_latency_is_int():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    assert isinstance(result["net_score_latency"], int)

# --- Value Ranges ---
def test_metrics_values_between_0_and_1():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    for k, v in result.items():
        if isinstance(v, float):
            assert 0.0 <= v <= 1.0

def test_metrics_size_score_is_dict():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    assert isinstance(result["size_score"], dict)

def test_metrics_size_score_keys():
    result = compute_metrics_for_model("https://huggingface.co/google/gemma-3-270m")
    expected_keys = {"raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"}
    assert expected_keys.issubset(result["size_score"].keys())
