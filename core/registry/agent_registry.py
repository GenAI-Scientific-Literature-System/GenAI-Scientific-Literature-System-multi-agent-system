"""
Agent Registry for Multi-Agent Architecture.
"""
from typing import Dict, Type, Any, Optional, List


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, Type[Any]] = {}

    def register(self, cls_or_name: Any = None):
        """Decorator or direct call to register an agent class."""
        def decorator(cls: Type[Any]) -> Type[Any]:
            name = getattr(cls, "agent_id", cls.__name__)
            self._agents[name] = cls
            return cls

        if callable(cls_or_name):
            return decorator(cls_or_name)
        return decorator

    def get(self, name: str) -> Optional[Type[Any]]:
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())


registry = AgentRegistry()
