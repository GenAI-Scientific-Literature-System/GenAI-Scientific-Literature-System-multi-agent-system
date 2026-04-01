import time
from typing import Dict, Any


class ExecutionMonitor:
    def __init__(self):
        self.records = {}

    def start(self, agent_name: str):
        self.records[agent_name] = {
            "start_time": time.time(),
            "end_time": None,
            "duration_sec": None,
            "status": "running",
            "error": None
        }

    def stop(self, agent_name: str, status: str = "success", error: str = None):
        if agent_name not in self.records:
            return

        self.records[agent_name]["end_time"] = time.time()
        self.records[agent_name]["duration_sec"] = round(
            self.records[agent_name]["end_time"] - self.records[agent_name]["start_time"], 4
        )
        self.records[agent_name]["status"] = status
        self.records[agent_name]["error"] = error

    def get(self, agent_name: str) -> Dict[str, Any]:
        return self.records.get(agent_name, {})

    def all(self) -> Dict[str, Any]:
        return self.records