import sys
import json
from pathlib import Path
from metrics import compute_metrics_for_model

def main(url_file: str) -> None:
    path = Path(url_file)
    if not path.exists():
        print(f"Error: File not found: {url_file}", file=sys.stderr)
        sys.exit(1)

    urls = path.read_text().splitlines()
    for url in urls:
        if "huggingface.co/" in url and "/datasets/" not in url:
            result = compute_metrics_for_model(url)
            print(json.dumps(result))
