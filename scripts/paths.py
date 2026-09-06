"""Repository-relative configuration paths and compatibility for relocated v0 inputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def processed_path(value):
    """Resolve the old flat Kepler v0 layout without changing saved run records."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if path.parent.parts[-3:] == ("data", "processed", "kepler") and "_v0" in path.name:
        moved = path.parent / "v0" / path.name
        if moved.is_file():
            return moved
    return path


def configuration_paths(config):
    """Resolve portable input/output paths relative to this repository, not the shell cwd."""
    result = dict(config)
    for key in ("base_dir", "dataset_path", "splits_path", "manifest_path", "output_root",
                "cnn_run_dir", "hgb_run_dir"):
        if result.get(key):
            path = Path(result[key]).expanduser()
            result[key] = str(processed_path(path if path.is_absolute() else ROOT / path).resolve())
    return result
