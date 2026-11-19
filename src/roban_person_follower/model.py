import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


MODEL_NAME = "yolox_nano.onnx"
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/" + MODEL_NAME
MODEL_SHA256 = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"


def ros_data_dir() -> Path:
    return Path(os.environ.get("ROS_HOME") or Path.home() / ".ros").expanduser() / "roban_person_follower"


def default_model_path() -> Path:
    return ros_data_dir() / "models" / MODEL_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(destination: Path) -> Path:
    destination = destination.expanduser()
    if destination.exists() and sha256(destination) == MODEL_SHA256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary_file = Path(output.name)
            with urlopen(MODEL_URL, timeout=30) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        if sha256(temporary_file) != MODEL_SHA256:
            raise ValueError("YOLOX-Nano download failed SHA-256 verification")
        temporary_file.replace(destination)
    finally:
        if temporary_file is not None and temporary_file.exists():
            temporary_file.unlink()
    return destination
