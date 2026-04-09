from typing import Dict, Any


def compute_system_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    execution = result.get("execution", {})
    errors = result.get("errors", [])

    total_agents = len(execution)
    successful_agents = sum(1 for v in execution.values() if v.get("status") == "success")
    failed_agents = sum(1 for v in execution.values() if v.get("status") == "failed")
    total_time = round(sum(v.get("duration_sec", 0) or 0 for v in execution.values()), 4)

    return {
        "total_agents_run": total_agents,
        "successful_agents": successful_agents,
        "failed_agents": failed_agents,
        "error_count": len(errors),
        "total_execution_time_sec": total_time
    }