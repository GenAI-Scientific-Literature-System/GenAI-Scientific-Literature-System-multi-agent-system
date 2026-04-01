def build_performance_monitor_view(result: dict) -> dict:
    execution = result.get("execution", {})
    performance_summary = result.get("performance_summary", {})

    total_time = 0.0
    slowest_agent = None
    slowest_time = -1.0

    agent_timings = []

    for agent_name, stats in execution.items():
        duration = stats.get("duration_sec", 0) or 0
        total_time += duration

        if duration > slowest_time:
            slowest_time = duration
            slowest_agent = agent_name

        agent_timings.append({
            "agent": agent_name,
            "duration_sec": duration,
            "status": stats.get("status", "unknown")
        })

    average_time = 0.0
    if execution:
        average_time = total_time / len(execution)

    return {
        "total_execution_time_sec": round(total_time, 4),
        "average_agent_time_sec": round(average_time, 4),
        "slowest_agent": slowest_agent,
        "slowest_agent_time_sec": round(slowest_time, 4) if slowest_time >= 0 else 0,
        "agent_timings": agent_timings,
        "performance_summary": performance_summary
    }