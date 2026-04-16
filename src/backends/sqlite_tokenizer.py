"""SQLite-backed tokenizer that mimics Qwen2 tokenizer interface.

Stores vocabulary in a SQLite database with indexed columns for fast
encode (token text → token_id) and decode (token_id → token text) lookups.
The connection is long-lived and uses WAL journal mode so that multiple
read-only processes can share the same database file concurrently.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Mock Qwen2 vocabulary (vocab_size = 100)
# ---------------------------------------------------------------------------
# Layout:
#   0-3   : special tokens  (<|endoftext|>, <|im_start|>, <|im_end|>, <pad>)
#   4     : newline character
#   5-99  : printable ASCII 0x20 (' ') .. 0x7E ('~'), 95 characters
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS: List[Tuple[int, str]] = [
    (0, "<|endoftext|>"),
    (1, "<|im_start|>"),
    (2, "<|im_end|>"),
    (3, "<pad>"),
]

_NUM_SPECIAL = len(_SPECIAL_TOKENS)
_NEWLINE_ID = _NUM_SPECIAL          # 4
_ASCII_START_ID = _NUM_SPECIAL + 1  # 5
_ASCII_CHAR_START = 0x20            # ' '
_ASCII_CHAR_END = 0x7E              # '~'


def _build_qwen2_vocab() -> List[Tuple[int, str, int]]:
    """Return list of ``(token_id, token_text, is_special)`` triples.

    Produces exactly 100 entries matching the tiny-Qwen2 vocab_size.
    """
    rows: List[Tuple[int, str, int]] = []

    # Special tokens
    for tid, text in _SPECIAL_TOKENS:
        rows.append((tid, text, 1))

    # Newline
    rows.append((_NEWLINE_ID, "\n", 0))

    # Printable ASCII characters
    tid = _ASCII_START_ID
    for code in range(_ASCII_CHAR_START, _ASCII_CHAR_END + 1):
        rows.append((tid, chr(code), 0))
        tid += 1

    assert len(rows) == 100, f"Expected 100 tokens, got {len(rows)}"
    return rows


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS vocab (
    token_id   INTEGER PRIMARY KEY,
    token      TEXT    NOT NULL,
    is_special INTEGER NOT NULL DEFAULT 0
);
"""

_INDEX_TOKEN_SQL = """\
CREATE INDEX IF NOT EXISTS idx_vocab_token ON vocab(token);
"""

# ---------------------------------------------------------------------------
# SqliteTokenizer
# ---------------------------------------------------------------------------


class SqliteTokenizer:
    """Character-level tokenizer backed by a SQLite vocabulary table.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Use ``":memory:"`` for an
        ephemeral in-memory database (default).
    vocab : list of (token_id, token_text, is_special), optional
        If provided, the vocabulary is (re-)populated on construction.
        When *None* the default mock-Qwen2 vocabulary is used.

    Notes
    -----
    * ``token_id`` column is ``INTEGER PRIMARY KEY`` → automatic B-tree
      index (used by ``decode``).
    * An additional index on ``token`` enables O(log N) lookups for
      ``encode``.
    * The connection uses WAL journal mode so multiple readers can
      co-exist with one writer, making it safe for multi-process
      scenarios where one process writes and others only read.
    """

    # Qwen2 compatible attributes
    eos_token_id: int = 0          # <|endoftext|>
    im_start_token_id: int = 1     # <|im_start|>
    im_end_token_id: int = 2       # <|im_end|>
    pad_token_id: int = 3          # <pad>

    def __init__(
        self,
        db_path: str = ":memory:",
        vocab: Optional[List[Tuple[int, str, int]]] = None,
    ) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        # Create schema
        self._conn.execute(_SCHEMA_SQL)
        self._conn.execute(_INDEX_TOKEN_SQL)
        self._conn.commit()

        # Populate vocabulary if table is empty
        row = self._conn.execute("SELECT COUNT(*) FROM vocab").fetchone()
        if row[0] == 0:
            entries = vocab if vocab is not None else _build_qwen2_vocab()
            self._conn.executemany(
                "INSERT INTO vocab (token_id, token, is_special) VALUES (?, ?, ?)",
                entries,
            )
            self._conn.commit()

        # Build in-memory caches for fast Python-side lookups
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._special_tokens: Dict[str, int] = {}  # text → id, longest first
        self._vocab_size: int = 0
        self._load_caches()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_caches(self) -> None:
        """Pre-load id↔token mappings from SQLite into Python dicts."""
        rows = self._conn.execute(
            "SELECT token_id, token, is_special FROM vocab"
        ).fetchall()
        for tid, text, is_special in rows:
            self._id_to_token[tid] = text
            self._token_to_id[text] = tid
            if is_special:
                self._special_tokens[text] = tid

        self._vocab_size = len(rows)

        # Sort special tokens by descending length for greedy matching
        self._special_sorted: List[Tuple[str, int]] = sorted(
            self._special_tokens.items(), key=lambda x: len(x[0]), reverse=True
        )

    # ------------------------------------------------------------------
    # Public API — encode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode *text* into a list of token ids.

        Uses greedy longest-match for special tokens, then character-level
        tokenization for ordinary characters.  Unknown characters are
        silently skipped (matching Qwen2 behaviour for out-of-vocab bytes).

        Parameters
        ----------
        text : str
            Input string.
        add_special_tokens : bool
            If *True*, prepend ``<|im_start|>`` and append ``<|im_end|>``
            around the encoded ids (ChatML-style).
        """
        ids: List[int] = []
        pos = 0
        n = len(text)

        while pos < n:
            matched = False
            # Try special tokens (longest first)
            for sp_text, sp_id in self._special_sorted:
                if text[pos: pos + len(sp_text)] == sp_text:
                    ids.append(sp_id)
                    pos += len(sp_text)
                    matched = True
                    break
            if matched:
                continue

            # Single character lookup
            ch = text[pos]
            tid = self._token_to_id.get(ch)
            if tid is not None:
                ids.append(tid)
            # else: skip unknown character
            pos += 1

        if add_special_tokens:
            ids = [self.im_start_token_id] + ids + [self.im_end_token_id]

        return ids

    # ------------------------------------------------------------------
    # Public API — decode
    # ------------------------------------------------------------------

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a sequence of token ids back to a string.

        Parameters
        ----------
        token_ids : sequence of int
            Token ids to decode.
        skip_special_tokens : bool
            When *True*, special tokens are omitted from the output.
        """
        parts: List[str] = []
        for tid in token_ids:
            text = self._id_to_token.get(tid)
            if text is None:
                continue  # unknown id
            if skip_special_tokens and tid in (
                self.eos_token_id,
                self.im_start_token_id,
                self.im_end_token_id,
                self.pad_token_id,
            ):
                continue
            parts.append(text)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Public API — vocabulary info
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (cached at load time)."""
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Return full vocabulary as ``{token_text: token_id}`` dict."""
        return dict(self._token_to_id)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Look up a single token id → text (or *None*)."""
        return self._id_to_token.get(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        """Look up a single token text → id (or *None*)."""
        return self._token_to_id.get(token)

    # ------------------------------------------------------------------
    # Connection / lifecycle
    # ------------------------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying long-lived SQLite connection."""
        return self._conn

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Factory — from pretrained HuggingFace tokenizer
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        db_path: str = ":memory:",
    ) -> "SqliteTokenizer":
        """Create a ``SqliteTokenizer`` with vocabulary from a HuggingFace tokenizer.

        Loads the full BPE vocabulary (including special tokens) via
        ``AutoTokenizer.from_pretrained`` and writes every entry into the
        SQLite ``vocab`` table.

        Parameters
        ----------
        model_name_or_path : str
            HuggingFace model identifier or local path, e.g.
            ``"Qwen/Qwen2.5-0.5B-Instruct"``.
        db_path : str
            SQLite database path (default ``":memory:"``).

        Returns
        -------
        SqliteTokenizer
            A tokenizer whose ``vocab`` table mirrors the HuggingFace
            vocabulary.
        """
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Build (token_id, token_text, is_special) triples
        all_special: set = set(hf_tok.all_special_tokens)
        vocab_map = hf_tok.get_vocab()          # {text: id}
        entries: List[Tuple[int, str, int]] = [
            (tid, text, int(text in all_special))
            for text, tid in vocab_map.items()
        ]

        instance = cls(db_path=db_path, vocab=entries)

        # Align special-token id attributes with the real tokenizer
        if hf_tok.eos_token_id is not None:
            instance.eos_token_id = hf_tok.eos_token_id
        if hasattr(hf_tok, "pad_token_id") and hf_tok.pad_token_id is not None:
            instance.pad_token_id = hf_tok.pad_token_id

        return instance

    def __repr__(self) -> str:
        return (
            f"SqliteTokenizer(db_path={self._db_path!r}, "
            f"vocab_size={self.vocab_size})"
        )
