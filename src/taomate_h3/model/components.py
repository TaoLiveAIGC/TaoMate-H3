# Modified for TaoMate-H3 streaming inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Load the VAE components bundled with MiniMax H3 weights."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


class H3ComponentError(RuntimeError):
    pass


_COMPONENT_PACKAGE_NAMES: dict[Path, str] = {}


def partition_root(model_root: str | Path) -> Path:
    root = (Path(model_root).expanduser() / "FL2VA").resolve(strict=True)
    if not root.is_dir():
        raise H3ComponentError(f"H3 partition directory is unavailable: {root}")
    return root


def read_component_config(component_root: str | Path) -> dict[str, Any]:
    root = Path(component_root).resolve(strict=True)
    path = root / "config.json"
    if not path.is_file():
        raise H3ComponentError(f"H3 component config is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3ComponentError(f"cannot load H3 component config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise H3ComponentError(f"H3 component config must be an object: {path}")
    return value


def _component_package(component_root: Path, entry_name: str) -> types.ModuleType:
    root = component_root.resolve(strict=True)
    if not root.is_dir():
        raise H3ComponentError("H3 component root must be a directory")
    entry = root / entry_name
    if not entry.is_file():
        raise H3ComponentError(f"H3 component entry is unavailable: {entry}")
    package_name = _COMPONENT_PACKAGE_NAMES.setdefault(
        root,
        f"_taomate_h3_component_{len(_COMPONENT_PACKAGE_NAMES)}",
    )
    module_name = f"{package_name}.{entry.stem}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__file__ = str(root / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, entry)
    if spec is None or spec.loader is None:
        raise H3ComponentError(f"cannot create module spec for {entry}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _configure_video_vae_tile_parallel(
    module: types.ModuleType,
    *,
    group: Any | None,
    rank: int,
    world_size: int,
) -> None:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(
            f"invalid video VAE tile-parallel topology: world_size={world_size}, rank={rank}"
        )
    if world_size > 1 and group is None:
        raise ValueError("video VAE tile-parallel group is required")
    get_state = getattr(module, "get_parallel_state", None)
    if not callable(get_state):
        raise H3ComponentError("bundled video VAE parallel state is unavailable")
    state = get_state()
    if not isinstance(state, dict):
        raise H3ComponentError("bundled video VAE parallel state must be a dict")
    state.clear()
    state.update(
        {
            "group_size": world_size,
            "group_rank": rank,
            "local_process_group": group,
            "sp_size": world_size,
            "sp_rank": rank,
            # The checkpoint enables parallel_tiling.  Keep the separate
            # convolution/attention spatial-parallel path disabled so each
            # rank independently encodes its assigned tiles.
            "sp_enabled": False,
            "sp_process_group": group,
            "tp_size": 1,
            "tp_rank": 0,
        }
    )


def configure_bundled_video_vae_tile_parallel(
    video_vae: Any,
    *,
    group: Any | None,
    rank: int,
    world_size: int,
) -> None:
    """Switch the bundled VAE between spatial tile parallel and rank-local tiles."""

    module = sys.modules.get(type(video_vae).__module__)
    if module is None:
        raise H3ComponentError("bundled video VAE module is unavailable")
    _configure_video_vae_tile_parallel(
        module,
        group=group,
        rank=rank,
        world_size=world_size,
    )


def load_bundled_video_vae(
    component_root: str | Path,
    *,
    tile_parallel_group: Any | None = None,
    tile_parallel_rank: int = 0,
    tile_parallel_world_size: int = 1,
) -> Any:
    root = Path(component_root).resolve(strict=True)
    module = _component_package(root, "minimax_h3_video_vae.py")
    _configure_video_vae_tile_parallel(
        module,
        group=tile_parallel_group,
        rank=tile_parallel_rank,
        world_size=tile_parallel_world_size,
    )
    cls = getattr(module, "MiniMaxH3VideoVAE", None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise H3ComponentError("bundled MiniMaxH3VideoVAE entry is invalid")
    return cls.from_pretrained(str(root))


def load_bundled_audio_vae(component_root: str | Path) -> Any:
    root = Path(component_root).resolve(strict=True)
    module = _component_package(root, "minimax_h3_audio_vae.py")
    cls = getattr(module, "MiniMaxH3AudioVAE", None)
    if cls is None or not callable(getattr(cls, "from_pretrained", None)):
        raise H3ComponentError("bundled MiniMaxH3AudioVAE entry is invalid")
    return cls.from_pretrained(str(root))


__all__ = [
    "configure_bundled_video_vae_tile_parallel",
    "H3ComponentError",
    "load_bundled_audio_vae",
    "load_bundled_video_vae",
    "partition_root",
    "read_component_config",
]
