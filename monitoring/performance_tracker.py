class PerformanceTracker:
    def __init__(self):
        self.data = {}

    def record(self, agent_name: str, duration_sec: float, success: bool):
        if agent_name not in self.data:
            self.data[agent_name] = {
                "runs": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_duration_sec": 0.0
            }

        self.data[agent_name]["runs"] += 1
        self.data[agent_name]["total_duration_sec"] += duration_sec or 0.0

        if success:
            self.data[agent_name]["success_count"] += 1
        else:
            self.data[agent_name]["failure_count"] += 1

    def summary(self):
        summary = {}
        for agent_name, stats in self.data.items():
            runs = stats["runs"]
            avg_time = stats["total_duration_sec"] / runs if runs else 0.0
            summary[agent_name] = {
                "runs": runs,
                "success_count": stats["success_count"],
                "failure_count": stats["failure_count"],
                "average_duration_sec": round(avg_time, 4)
            }
        return summary