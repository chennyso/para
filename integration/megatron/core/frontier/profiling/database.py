"""
Profile database — SQLite-backed store of region and transition timing/memory.

Schema:
  region_profile:     profile_id, gpu, dtype, seq_len, micro_batch, hidden,
                      tp, cp, ep, pp, forward_us_p50, backward_us_p50,
                      peak_memory_bytes, forward_us_p10, forward_us_p90
  transition_profile: lowering_id, obligation_kind, shape_sig, mesh_id,
                      forward_us_p50, backward_us_p50, bytes, temp_memory_bytes
  contention:         resource_a, resource_b, factor_p50

All measurements use 10 warmup + 50 measured, median reported.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RegionProfile:
    profile_id: str
    forward_us: float       # P50
    backward_us: float      # P50
    peak_memory_bytes: int
    forward_us_p10: float = 0.0
    forward_us_p90: float = 0.0


@dataclass
class TransitionProfile:
    lowering_id: str
    forward_us: float
    backward_us: float
    bytes: int
    temp_memory_bytes: int = 0


class ProfileDatabase:
    """
    Thread-safe read, single-writer profile database.
    """

    def __init__(self, path: str = "profiles/frontier.db"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self._conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS region_profile (
                profile_id TEXT,
                gpu TEXT,
                dtype TEXT,
                seq_len INTEGER,
                micro_batch INTEGER,
                hidden INTEGER,
                tp INTEGER,
                cp INTEGER,
                ep INTEGER,
                pp INTEGER,
                forward_us_p50 REAL,
                backward_us_p50 REAL,
                peak_memory_bytes INTEGER,
                forward_us_p10 REAL DEFAULT 0,
                forward_us_p90 REAL DEFAULT 0,
                PRIMARY KEY (profile_id, gpu, dtype, seq_len, micro_batch, hidden, tp, cp, ep, pp)
            );

            CREATE TABLE IF NOT EXISTS transition_profile (
                lowering_id TEXT,
                obligation_kind TEXT,
                shape_sig TEXT,
                mesh_id TEXT,
                forward_us_p50 REAL,
                backward_us_p50 REAL,
                bytes INTEGER,
                temp_memory_bytes INTEGER DEFAULT 0,
                PRIMARY KEY (lowering_id, obligation_kind, shape_sig, mesh_id)
            );

            CREATE TABLE IF NOT EXISTS contention (
                resource_a TEXT,
                resource_b TEXT,
                factor_p50 REAL,
                PRIMARY KEY (resource_a, resource_b)
            );
        """)
        self._conn.commit()

    def insert_region(
        self,
        profile_id: str,
        forward_us: float,
        backward_us: float,
        peak_memory_bytes: int,
        gpu: str = "h100",
        dtype: str = "bf16",
        seq_len: int = 2048,
        micro_batch: int = 1,
        hidden: int = 4096,
        tp: int = 1,
        cp: int = 1,
        ep: int = 1,
        pp: int = 1,
        fwd_p10: float = 0.0,
        fwd_p90: float = 0.0,
    ):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO region_profile
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (profile_id, gpu, dtype, seq_len, micro_batch, hidden,
             tp, cp, ep, pp, forward_us, backward_us, peak_memory_bytes,
             fwd_p10, fwd_p90),
        )
        self._conn.commit()

    def lookup_region(self, profile_id: str) -> Optional[RegionProfile]:
        row = self._conn.execute(
            "SELECT forward_us_p50, backward_us_p50, peak_memory_bytes, "
            "forward_us_p10, forward_us_p90 "
            "FROM region_profile WHERE profile_id=? "
            "ORDER BY forward_us_p50 LIMIT 1",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        return RegionProfile(
            profile_id=profile_id,
            forward_us=row[0],
            backward_us=row[1],
            peak_memory_bytes=row[2],
            forward_us_p10=row[3],
            forward_us_p90=row[4],
        )

    def insert_transition(
        self,
        lowering_id: str,
        obligation_kind: str,
        shape_sig: tuple,
        mesh_id: str,
        forward_us: float,
        backward_us: float,
        bytes_count: int,
        temp_memory_bytes: int = 0,
    ):
        self._conn.execute(
            "INSERT OR REPLACE INTO transition_profile VALUES (?,?,?,?,?,?,?,?)",
            (lowering_id, obligation_kind, json.dumps(shape_sig), mesh_id,
             forward_us, backward_us, bytes_count, temp_memory_bytes),
        )
        self._conn.commit()

    def lookup_transition_us(self, key: tuple) -> Optional[float]:
        """key = (lowering_id, obligation_kind, shape, mesh_id)"""
        if len(key) < 4:
            return None
        row = self._conn.execute(
            "SELECT forward_us_p50 FROM transition_profile "
            "WHERE lowering_id=? AND obligation_kind=? AND shape_sig=? AND mesh_id=?",
            (key[0], key[1], json.dumps(key[2]), key[3]),
        ).fetchone()
        return row[0] if row else None

    def insert_contention(self, resource_a: str, resource_b: str, factor: float):
        self._conn.execute(
            "INSERT OR REPLACE INTO contention VALUES (?,?,?)",
            (resource_a, resource_b, factor),
        )
        self._conn.commit()

    def lookup_contention(self, resource_a: str, resource_b: str) -> float:
        row = self._conn.execute(
            "SELECT factor_p50 FROM contention WHERE resource_a=? AND resource_b=?",
            (resource_a, resource_b),
        ).fetchone()
        return row[0] if row else 1.0

    def close(self):
        self._conn.close()
