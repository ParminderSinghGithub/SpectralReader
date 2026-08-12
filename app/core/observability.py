import time
import uuid
from typing import Dict, Any, Optional
from contextlib import contextmanager

class StageTracker:
    """Tracks latency across pipeline stages."""
    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = time.perf_counter()
        self.stages: Dict[str, float] = {}

    @contextmanager
    def measure_stage(self, stage_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.stages[stage_name] = elapsed_ms

    def total_elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "total_elapsed_ms": self.total_elapsed_ms(),
            "stage_latencies_ms": self.stages
        }
