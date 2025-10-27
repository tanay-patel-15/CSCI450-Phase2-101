import json
import pytest
from src.cli import main

# --- Valid Model URL ---
def test_cli_outputs_json_for_model(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://huggingface.co/google/gemma-3-270m\n")
    main(str(urls_file))
    captured = capsys.readouterr()
    obj = json.loads(captured.out.strip())
    assert obj["category"] == "MODEL"

# --- Skips Dataset ---
def test_cli_skips_dataset(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://huggingface.co/datasets/xlangai/AgentNet\n")
    main(str(urls_file))
    captured = capsys.readouterr()
    assert captured.out.strip() == ""

# --- Multiple URLs ---
def test_cli_multiple_urls(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://huggingface.co/google/gemma-3-270m\n"
        "https://huggingface.co/datasets/xlangai/AgentNet\n"
    )
    main(str(urls_file))
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 1  # only the model

# --- Invalid File ---
def test_cli_missing_file(tmp_path, capsys):
    bad_file = tmp_path / "does_not_exist.txt"
    with pytest.raises(SystemExit):
        main(str(bad_file))
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err

# --- Empty File ---
def test_cli_empty_file(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("")
    main(str(urls_file))
    captured = capsys.readouterr()
    assert captured.out.strip() == ""

# --- Invalid URL ---
def test_cli_invalid_url(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("not_a_real_url\n")
    main(str(urls_file))
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # no crash


# --- Non-Model URL ---
def test_cli_non_model_url(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://github.com/SkyworkAI/Matrix-Game\n")
    main(str(urls_file))
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # ignored since not a model
 

# --- Multiple Models ---
def test_cli_multiple_models(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text(
        "https://huggingface.co/google/model1\n"
        "https://huggingface.co/google/model2\n"
    )
    main(str(urls_file))
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    assert all(o["category"] == "MODEL" for o in objs)

# --- NDJSON Validity ---
def test_cli_ndjson_output(tmp_path, capsys):
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://huggingface.co/google/gemma-3-270m\n")
    main(str(urls_file))
    captured = capsys.readouterr()
    for line in captured.out.strip().splitlines():
        json.loads(line)  # must be valid JSON
