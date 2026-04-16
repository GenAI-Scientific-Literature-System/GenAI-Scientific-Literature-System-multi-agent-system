import json

from orchestration.execution_monitor import ExecutionMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.metrics import compute_system_metrics
from monitoring.logger import get_logger


def test_execution_monitor():
    monitor = ExecutionMonitor()

    monitor.start("agent_1")
    monitor.stop("agent_1", status="success")

    result = monitor.get("agent_1")

    assert result["status"] == "success"
    assert result["duration_sec"] is not None
    print("ExecutionMonitor test passed")


def test_performance_tracker():
    tracker = PerformanceTracker()

    tracker.record("agent_1", 1.2, True)
    tracker.record("agent_1", 0.8, True)
    tracker.record("agent_2", 2.0, False)

    summary = tracker.summary()

    assert summary["agent_1"]["runs"] == 2
    assert summary["agent_1"]["success_count"] == 2
    assert summary["agent_2"]["failure_count"] == 1
    print("PerformanceTracker test passed")


def test_system_metrics():
    sample_result = {
        "execution": {
            "agent_1": {"status": "success", "duration_sec": 1.1},
            "agent_2": {"status": "failed", "duration_sec": 0.7},
            "agent_3": {"status": "success", "duration_sec": 2.2},
        },
        "errors": [
            {"agent": "agent_2", "error": "Sample failure"}
        ]
    }

    metrics = compute_system_metrics(sample_result)

    assert metrics["total_agents_run"] == 3
    assert metrics["successful_agents"] == 2
    assert metrics["failed_agents"] == 1
    assert metrics["error_count"] == 1
    print("System metrics test passed")


def test_logger():
    logger = get_logger("test_logger")
    logger.info("Logger test message")
    print("Logger test passed")


if __name__ == "__main__":
    test_execution_monitor()
    test_performance_tracker()
    test_system_metrics()
    test_logger()

    print("\nAll orchestration/monitoring tests passed")