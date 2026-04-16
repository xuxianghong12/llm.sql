from __future__ import annotations

from dataclasses import dataclass
import json
import os
import resource
import sqlite3
import sys
from typing import Any, Literal, Optional

from src.backends.llm_pcache import LLMPcache
from src.backends.standalone_engine import StandaloneEngine


SessionMode = Literal["single_sql", "eager", "layer_stream"]
PcachePolicy = Literal["lru", "opt"]
_DEFAULT_LAYER_STREAM_PCACHE_MB = 200.0


@dataclass
class SessionConfig:
    export_dir: str
    mode: SessionMode = "single_sql"
    db_filename: Optional[str] = None
    num_threads: int = 4
    pcache_budget_mb: Optional[float] = None
    use_pcache: bool = False
    pcache_policy: PcachePolicy = "opt"
    pcache_prefetch_window: int = 2
    warm_full_params: bool = False


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _read_manifest(export_dir: str) -> dict:
    path = os.path.join(export_dir, "manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _db_has_runtime_schema(export_dir: str, db_filename: str) -> bool:
    db_path = os.path.join(export_dir, db_filename)
    if not os.path.isfile(db_path):
        return False

    required_tables = {"metadata", "model_params"}
    if not os.path.isfile(os.path.join(export_dir, "tokenizer.json")):
        required_tables.add("vocab")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False

    try:
        tables = {
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    except sqlite3.Error:
        return False
    finally:
        conn.close()

    return required_tables.issubset(tables)


def _resolve_db_filename(export_dir: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    if _db_has_runtime_schema(export_dir, "model.db"):
        return "model.db"

    manifest = _read_manifest(export_dir)
    default_db = manifest.get("default_db")
    if isinstance(default_db, str) and _db_has_runtime_schema(export_dir, default_db):
            return default_db

    if _db_has_runtime_schema(export_dir, "model_int8.db"):
        return "model_int8.db"

    model_db = os.path.join(export_dir, "model.db")
    if os.path.isfile(model_db):
        return "model.db"
    return "model.db"


def _resolve_num_layers(export_dir: str) -> int:
    manifest = _read_manifest(export_dir)
    value = manifest.get("num_layers")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


class InferenceSession:
    def __init__(self, config: SessionConfig) -> None:
        self._config = config
        self._pcache: Optional[LLMPcache] = None

        db_filename = _resolve_db_filename(config.export_dir, config.db_filename)
        self._db_filename = db_filename
        num_layers = _resolve_num_layers(config.export_dir)

        lazy_params = config.mode == "layer_stream"
        layer_stream_low_memory = lazy_params
        use_pcache = (
            config.use_pcache
            or config.pcache_budget_mb is not None
            or layer_stream_low_memory
        )
        if layer_stream_low_memory:
            budget_mb = (
                config.pcache_budget_mb
                if config.pcache_budget_mb is not None
                else _DEFAULT_LAYER_STREAM_PCACHE_MB
            )
        else:
            budget_mb = (
                config.pcache_budget_mb
                if config.pcache_budget_mb is not None
                else 512.0
            )

        if use_pcache:
            self._pcache = LLMPcache(
                budget_mb=budget_mb,
                num_layers=num_layers,
                policy=config.pcache_policy,
                prefetch_window=config.pcache_prefetch_window,
            )
            self._pcache.install()

        single_sql_mode = config.mode == "single_sql" or layer_stream_low_memory
        disable_mmap = lazy_params

        self._engine = StandaloneEngine(
            config.export_dir,
            db_filename=db_filename,
            pcache=self._pcache,
            lazy_params=lazy_params,
            disable_mmap=disable_mmap,
            single_sql_mode=single_sql_mode,
        )
        self._engine.set_threads(config.num_threads)

        if config.mode == "eager" or config.warm_full_params:
            self._touch_all_params()

    @property
    def engine(self) -> StandaloneEngine:
        return self._engine

    @property
    def db_filename(self) -> str:
        return self._db_filename

    @property
    def last_profile(self) -> Optional[dict[str, Any]]:
        profile = getattr(self._engine, "last_profile", None)
        if isinstance(profile, dict):
            return profile
        return None

    def _touch_all_params(self) -> int:
        total = 0
        for (blob,) in self._engine.connection.execute("SELECT data FROM model_params"):
            total += len(blob)
        return total

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        verbose: bool = False,
        profile: bool = False,
    ) -> str:
        text = self._engine.generate(
            prompt,
            max_tokens=max_tokens,
            verbose=verbose,
            profile=profile,
        )
        if profile and isinstance(getattr(self._engine, "last_profile", None), dict):
            self._engine.last_profile["peak_rss_mb"] = round(_peak_rss_mb(), 1)
            self._engine.last_profile["mode"] = self._config.mode
            self._engine.last_profile["db_filename"] = self._db_filename
        return text

    def close(self) -> None:
        try:
            self._engine.close()
        finally:
            if self._pcache is not None:
                self._pcache.uninstall()
                self._pcache = None

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def generate_text(
    export_dir: str,
    prompt: str,
    max_tokens: int = 64,
    mode: SessionMode = "single_sql",
    db_filename: Optional[str] = None,
    num_threads: int = 4,
    pcache_budget_mb: Optional[float] = None,
    use_pcache: bool = False,
    pcache_policy: PcachePolicy = "opt",
    pcache_prefetch_window: int = 2,
    verbose: bool = False,
) -> str:
    config = SessionConfig(
        export_dir=export_dir,
        mode=mode,
        db_filename=db_filename,
        num_threads=num_threads,
        pcache_budget_mb=pcache_budget_mb,
        use_pcache=use_pcache,
        pcache_policy=pcache_policy,
        pcache_prefetch_window=pcache_prefetch_window,
    )
    with InferenceSession(config) as session:
        return session.generate(prompt, max_tokens=max_tokens, verbose=verbose)
