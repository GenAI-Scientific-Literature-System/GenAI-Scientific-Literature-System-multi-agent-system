def build_system_status_view(result: dict) -> dict:
    execution = result.get("execution", {})
    errors = result.get("errors", [])

    agent_status = []
    total_agents = 0
    successful_agents = 0
    failed_agents = 0

    for agent_name, stats in execution.items():
        total_agents += 1
        status = stats.get("status", "unknown")

        if status == "success":
            successful_agents += 1
        elif status == "failed":
            failed_agents += 1

        agent_status.append({
            "agent": agent_name,
            "status": status,
            "duration_sec": stats.get("duration_sec", 0),
            "error": stats.get("error")
        })

    overall_status = "healthy"
    if failed_agents > 0:
        overall_status = "degraded"
    if total_agents == 0:
        overall_status = "not_run"

    return {
        "overall_status": overall_status,
        "total_agents": total_agents,
        "successful_agents": successful_agents,
        "failed_agents": failed_agents,
        "errors": errors,
        "agent_status": agent_status
    }