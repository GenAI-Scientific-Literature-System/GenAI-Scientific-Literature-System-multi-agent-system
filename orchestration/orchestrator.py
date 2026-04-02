from typing import Dict, Any, List

from orchestration.pipeline_state import PipelineState
from orchestration.execution_monitor import ExecutionMonitor
from monitoring.logger import get_logger
from monitoring.performance_tracker import PerformanceTracker

from agents.claim_extraction import ClaimExtractor
from agents.evidence_collection import EvidenceCollector
from agents.reliability_analysis import ReliabilityAnalyzer
from agents.agreement_detection import AgreementDetector
from agents.uncertainty_detection import UncertaintyDetector
from agents.ranking_prioritization import RankingPrioritizer


class MultiAgentOrchestrator:
    def __init__(self):
        self.logger = get_logger("multi_agent_orchestrator")
        self.monitor = ExecutionMonitor()
        self.performance_tracker = PerformanceTracker()

        self.agent1 = ClaimExtractor()
        self.agent2 = EvidenceCollector()
        self.agent3 = ReliabilityAnalyzer()
        self.agent4 = AgreementDetector()
        self.agent5 = UncertaintyDetector()
        self.agent6 = RankingPrioritizer()

    def _run_agent(self, agent_name: str, agent_callable, *args, **kwargs):
        self.monitor.start(agent_name)
        self.logger.info(f"Starting {agent_name}")

        try:
            result = agent_callable(*args, **kwargs)
            self.monitor.stop(agent_name, status="success")
            self.performance_tracker.record(agent_name, self.monitor.get(agent_name)["duration_sec"], True)
            self.logger.info(f"Completed {agent_name}")
            return result

        except Exception as e:
            self.monitor.stop(agent_name, status="failed", error=str(e))
            self.performance_tracker.record(agent_name, self.monitor.get(agent_name).get("duration_sec", 0), False)
            self.logger.exception(f"{agent_name} failed: {str(e)}")
            raise

    def run(self, query: str, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        state = PipelineState(query=query, papers=papers)

        try:
            claims = self._run_agent("agent_1_claim_extraction", self.agent1.extract, papers)
            state.set_output("claims", claims)
            state.set_execution("agent_1_claim_extraction", self.monitor.get("agent_1_claim_extraction"))
        except Exception as e:
            state.add_error("agent_1_claim_extraction", str(e))
            claims = []

        try:
            evidence = self._run_agent("agent_2_evidence_collection", self.agent2.collect, claims, papers)
            state.set_output("evidence", evidence)
            state.set_execution("agent_2_evidence_collection", self.monitor.get("agent_2_evidence_collection"))
        except Exception as e:
            state.add_error("agent_2_evidence_collection", str(e))
            evidence = []

        try:
            reliability = self._run_agent("agent_3_reliability_analysis", self.agent3.batch_evaluate, papers)
            state.set_output("reliability", reliability)
            state.set_execution("agent_3_reliability_analysis", self.monitor.get("agent_3_reliability_analysis"))
        except Exception as e:
            state.add_error("agent_3_reliability_analysis", str(e))
            reliability = []

        try:
            agreements = self._run_agent("agent_4_agreement_detection", self.agent4.detect, claims)
            state.set_output("agreements", agreements)
            state.set_execution("agent_4_agreement_detection", self.monitor.get("agent_4_agreement_detection"))
        except Exception as e:
            state.add_error("agent_4_agreement_detection", str(e))
            agreements = []

        try:
            uncertainties = self._run_agent("agent_5_uncertainty_detection", self.agent5.detect, claims, evidence)
            state.set_output("uncertainties", uncertainties)
            state.set_execution("agent_5_uncertainty_detection", self.monitor.get("agent_5_uncertainty_detection"))
        except Exception as e:
            state.add_error("agent_5_uncertainty_detection", str(e))
            uncertainties = []

        try:
            ranked_insights = self._run_agent(
                "agent_6_ranking_prioritization",
                self.agent6.rank,
                claims,
                evidence,
                reliability,
                agreements,
                uncertainties
            )
            state.set_output("ranked_insights", ranked_insights)
            state.set_execution("agent_6_ranking_prioritization", self.monitor.get("agent_6_ranking_prioritization"))
        except Exception as e:
            state.add_error("agent_6_ranking_prioritization", str(e))
            ranked_insights = []

        final_output = {
            "query": query,
            "claims": claims,
            "evidence": evidence,
            "reliability": reliability,
            "agreements": agreements,
            "uncertainties": uncertainties,
            "ranked_insights": ranked_insights,
            "execution": self.monitor.all(),
            "performance_summary": self.performance_tracker.summary(),
            "errors": state.errors
        }

        return final_output