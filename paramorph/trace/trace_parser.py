"""
GPU Trace Parser - Parse Nsight Systems and PyTorch Profiler traces
"""
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sqlite3
import json
import heapq
from collections import defaultdict
from enum import Enum


class TraceEventType(Enum):
    CUDA_KERNEL = "cuda_kernel"
    NCCL_CALL = "nccl_call"
    NVTX_RANGE = "nvtx_range"
    CPU_OP = "cpu_op"
    CUDA_SYNC = "cuda_sync"


@dataclass
class TraceEvent:
    event_id: str
    event_type: TraceEventType
    rank: int
    start_ns: int
    end_ns: int
    duration_ns: int
    name: str
    stream_id: Optional[int] = None
    correlation_id: Optional[int] = None
    nvtx_domain: Optional[str] = None
    nvtx_category: Optional[str] = None
    kernel_args: Dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


@dataclass
class TraceBundle:
    """Complete trace data from all ranks"""
    events: List[TraceEvent]
    rank_count: int
    time_offset_ns: Dict[int, int]  # Per-rank time offset for synchronization
    source_type: str  # "nsys" or "pytorch"
    metadata: Dict = field(default_factory=dict)

    def get_rank_events(self, rank: int) -> List[TraceEvent]:
        return [e for e in self.events if e.rank == rank]

    def get_events_by_type(self, event_type: TraceEventType) -> List[TraceEvent]:
        return [e for e in self.events if e.event_type == event_type]


class TraceParser:
    """Parse GPU traces from multiple sources"""

    def __init__(self):
        self.next_event_id = 0

    def _gen_event_id(self) -> str:
        event_id = f"event_{self.next_event_id}"
        self.next_event_id += 1
        return event_id

    def parse_nsys_sqlite(self, db_path: Path, rank: int) -> List[TraceEvent]:
        """
        Parse Nsight Systems SQLite database

        Real implementation requires:
        - Connect to .nsys-rep or exported SQLite
        - Query CUPTI_ACTIVITY_KIND_KERNEL table for CUDA kernels
        - Query NVTX_EVENTS for NVTX ranges
        - Query for NCCL operations (may need custom markers)
        - Associate correlation IDs between CPU and GPU events
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Nsight Systems database not found: {db_path}")

        events = []

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check if required tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
                raise ValueError(
                    f"CUPTI_ACTIVITY_KIND_KERNEL table not found. "
                    f"Available tables: {tables}"
                )

            # Parse CUDA kernels
            cursor.execute("""
                SELECT k.correlationId, k.start, k.end, s.value, k.streamId
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                ORDER BY k.start
            """)

            for row in cursor.fetchall():
                corr_id, start_ns, end_ns, name, stream_id = row
                events.append(TraceEvent(
                    event_id=self._gen_event_id(),
                    event_type=TraceEventType.CUDA_KERNEL,
                    rank=rank,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    duration_ns=end_ns - start_ns,
                    name=name or "unknown_kernel",
                    stream_id=stream_id,
                    correlation_id=corr_id,
                ))

            # Parse NVTX ranges if available
            if "NVTX_EVENTS" in tables:
                cursor.execute("""
                    SELECT eventType, start, end, text
                    FROM NVTX_EVENTS
                    WHERE eventType IN (59, 60)
                    ORDER BY start
                """)

                for row in cursor.fetchall():
                    event_type, start_ns, end_ns, text = row
                    if start_ns and end_ns:
                        events.append(TraceEvent(
                            event_id=self._gen_event_id(),
                            event_type=TraceEventType.NVTX_RANGE,
                            rank=rank,
                            start_ns=start_ns,
                            end_ns=end_ns,
                            duration_ns=end_ns - start_ns,
                            name=text or "nvtx_range",
                        ))

            conn.close()

        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error parsing {db_path}: {e}")

        return events

    def parse_pytorch_trace(self, trace_path: Path, rank: int) -> List[TraceEvent]:
        """
        Parse PyTorch Profiler JSON trace

        Format: Chrome Trace Event Format
        """
        if not trace_path.exists():
            raise FileNotFoundError(f"PyTorch trace not found: {trace_path}")

        with open(trace_path) as f:
            trace_data = json.load(f)

        events = []

        # PyTorch profiler uses Chrome trace format
        if "traceEvents" in trace_data:
            for event in trace_data["traceEvents"]:
                if event.get("ph") in ["X", "B", "E"]:  # Complete or begin/end events
                    if "ts" in event and "dur" in event:
                        start_ns = int(event["ts"] * 1000)  # Convert µs to ns
                        dur_ns = int(event["dur"] * 1000)

                        # Determine event type from name
                        name = event.get("name", "")
                        if name.startswith('ParaMorph/') or event.get('cat') in ('user_annotation', 'nvtx'):
                            event_type = TraceEventType.NVTX_RANGE
                        elif event.get('cat') == 'kernel':
                            event_type = TraceEventType.NCCL_CALL if 'nccl' in name.lower() else TraceEventType.CUDA_KERNEL
                        elif event.get('cat') == 'cuda_runtime' and ('Synchronize' in name or 'WaitEvent' in name):
                            event_type = TraceEventType.CUDA_SYNC
                        else:
                            event_type = TraceEventType.CPU_OP

                        events.append(TraceEvent(
                            event_id=self._gen_event_id(),
                            event_type=event_type,
                            rank=rank,
                            start_ns=start_ns,
                            end_ns=start_ns + dur_ns,
                            duration_ns=dur_ns,
                            name=name,
                            stream_id=event.get("args", {}).get("stream"),
                            correlation_id=event.get('args', {}).get('correlation'),
                            kernel_args=dict(event.get('args', {}), cpu_thread=event.get('tid')),
                        ))

        return events

    def associate_nvtx_ranges(self, events: List[TraceEvent]) -> List[TraceEvent]:
        """
        Associate CUDA kernels with their NVTX ranges

        Matches based on time overlap
        """
        nvtx_ranges = [e for e in events if e.event_type == TraceEventType.NVTX_RANGE]
        cuda_kernels = [e for e in events if e.event_type in (TraceEventType.CUDA_KERNEL, TraceEventType.NCCL_CALL)]
        launches = {event.correlation_id: event for event in events
            if event.event_type in (TraceEventType.CPU_OP, TraceEventType.CUDA_SYNC)
            and event.correlation_id is not None}
        thread_ranges = defaultdict(list)
        thread_launches = defaultdict(list)
        for scope in nvtx_ranges:
            thread_ranges[scope.rank, scope.kernel_args.get('cpu_thread')].append(scope)
        for launch in launches.values():
            thread_launches[launch.rank, launch.kernel_args.get('cpu_thread')].append(launch)
        matched_scopes = {}
        for thread, local_launches in thread_launches.items():
            ranges = sorted(thread_ranges[thread], key=lambda item: item.start_ns)
            cursor = 0
            active = []
            for launch in sorted(local_launches, key=lambda item: item.start_ns):
                while cursor < len(ranges) and ranges[cursor].start_ns <= launch.start_ns:
                    scope = ranges[cursor]
                    heapq.heappush(active, (scope.duration_ns, scope.event_id, scope))
                    cursor += 1
                while active and active[0][2].end_ns < launch.start_ns:
                    heapq.heappop(active)
                if active:
                    matched_scopes[launch.event_id] = active[0][2]

        for kernel in cuda_kernels:
            launch = launches.get(kernel.correlation_id)
            if launch is None:
                kernel.kernel_args['identity_gap'] = '缺少对应CPU launch correlation'
                continue
            if launch.event_id in matched_scopes:
                scope = matched_scopes[launch.event_id]
                kernel.nvtx_domain = scope.name
                if scope.name.startswith('ParaMorph/{'):
                    kernel.kernel_args['paramorph_identity'] = json.loads(scope.name[len('ParaMorph/'):])
                kernel.kernel_args['launch_event_id'] = launch.event_id
            else:
                kernel.kernel_args['identity_gap'] = 'CPU launch缺少Region范围'

        return events

    def normalize_time_domains(
        self,
        rank_events: Dict[int, List[TraceEvent]],
        clock_calibration: Optional[Dict] = None,
    ) -> TraceBundle:
        """
        Normalize timestamps across ranks

        时间偏移必须来自实际跨rank时钟校准。
        """
        all_events = []
        time_offsets = {}

        if clock_calibration is None or set(clock_calibration) != set(rank_events):
            raise ValueError('跨rank时间比较必须提供实测时钟偏移和误差范围')
        for rank, events in rank_events.items():
            measurement = clock_calibration[rank]
            if measurement['uncertainty_ns'] < 0 or not measurement['evidence_id']:
                raise ValueError('时钟校准缺少证据或合法误差范围')
            time_offsets[rank] = measurement['offset_ns']

        # Normalize all events
        for rank, events in rank_events.items():
            offset = time_offsets[rank]
            for event in events:
                all_events.append(replace(event, start_ns=event.start_ns + offset,
                                          end_ns=event.end_ns + offset))

        return TraceBundle(
            events=all_events,
            rank_count=len(rank_events),
            time_offset_ns=time_offsets,
            source_type="mixed",
            metadata=dict(clock_calibration=clock_calibration),
        )

    def parse_multi_rank(
        self,
        trace_paths: Dict[int, Path],
        source_type: str = "nsys",
        clock_calibration: Optional[Dict] = None,
    ) -> TraceBundle:
        """
        Parse traces from multiple ranks

        Args:
            trace_paths: Dict mapping rank -> trace file path
            source_type: "nsys" or "pytorch"
        """
        rank_events = {}

        for rank, path in trace_paths.items():
            if source_type == "nsys":
                events = self.parse_nsys_sqlite(path, rank)
            elif source_type == "pytorch":
                events = self.parse_pytorch_trace(path, rank)
            else:
                raise ValueError(f"Unknown source type: {source_type}")

            # Associate NVTX ranges
            events = self.associate_nvtx_ranges(events)
            rank_events[rank] = events

        # Normalize time domains
        bundle = self.normalize_time_domains(rank_events, clock_calibration)
        bundle.source_type = source_type

        return bundle
