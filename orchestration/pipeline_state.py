from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class PipelineState:
    query: str
    papers: List[Dict[str, Any]]
    outputs: Dict[str, Any] = field(default_factory = dict)
    execution: Dict[str, Any] = field(default_factory = dict)
    errors: List[Dict[str, str]] = field(default_factory = list)

    def set_output(self, agent_name: str, output:Any):
        self.outputs[agent_name] = output

    def set_execution(self, agent_name:str, execution_info: Dict[str, Any]):
        self.execution[agent_name] = execution_info

    def add_error(self, agent_name: str, error_message: str):
        self.errors.append({"agent": agent_name, "error": error_message})

    def to_dict(self):
        return{
            "query": self.query,
            "papers": self.papers,
            "outputs": self.outputs,
            "execution": self.execution,
            "errors": self.errors
        }