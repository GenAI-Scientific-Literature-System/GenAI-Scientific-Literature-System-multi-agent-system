def test_orchestrator_initializes_without_optional_groq_keys(monkeypatch):
    for name in ["GROQ_API_KEY", *[f"GROQ_API_KEY_{i}" for i in range(1, 7)]]:
        monkeypatch.delenv(name, raising=False)

    from orchestration.orchestrator import MultiAgentOrchestrator

    orchestrator = MultiAgentOrchestrator()
    assert orchestrator.agent1 is not None
    assert orchestrator.agent2 is not None
