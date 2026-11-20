"""Get a pinned model. An installed model also works offline."""

import hashlib
import json
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from common import MODELS, ROOT


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_dir(name):
    lock = json.loads((ROOT / "data/models.json").read_text())
    model = lock[name]
    folder = MODELS / name
    for filename, expected in model["files"].items():
        path = folder / filename
        if path.is_file() and sha256(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name}/{filename}", flush=True)
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{filename}"
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                temporary_path = Path(temp.name)
            try:
                with urllib.request.urlopen(
                    f"{url}?download=true&v={time.time_ns()}", timeout=180
                ) as source:
                    with temporary_path.open("wb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                if sha256(temporary_path) != expected:
                    raise OSError(f"Checksum mismatch for {filename}")
                temporary_path.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
            finally:
                temporary_path.unlink(missing_ok=True)
    return folder
