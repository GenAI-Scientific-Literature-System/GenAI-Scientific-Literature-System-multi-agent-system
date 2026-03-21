"""
Epistemic Dependency Graph (EDG)  G = (V_C ∪ V_A, E)

ALL THREE ANALYTICAL LAYERS — fully computed, exposed to frontend:

A. COMMUNITY DETECTION (greedy modularity)
   clusters = greedy_modularity_communities(G)
   → Groups semantically related claims
   → Each cluster gets a theme label

B. INFLUENCE PROPAGATION (weighted, iterative)
   u[n] = 0.7*u[n] + 0.3*avg(weighted_neighbours)
   contradict edges amplify uncertainty (weight=1.4)
   agree edges dampen it (weight=0.8)
   → Uncertainty spreads realistically through the graph

C. REASONING PATHS
   For every contradiction pair (Ci, Cj):
     path = shortest_path(G, Ci, Cj)
   → C1 → A1 → C3 → Gap
   → Shows what chain of reasoning connects contradictions

Plus: betweenness, pagerank, clustering coefficient, gap regions.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from src.models.schemas import Claim, Agreement, Assumption

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx not installed.")


class EpistemicDependencyGraph:

    def __init__(self):
        self.G       = nx.DiGraph() if HAS_NX else None
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict]      = []

    # ── Construction ──────────────────────────────────────────────────────────

    def add_claim_node(self, claim: Claim):
        data = {
            "type": "claim", "text": claim.text, "domain": claim.domain,
            "method": claim.method, "uncertainty": claim.uncertainty,
            "paper_id": claim.paper_id,
        }
        self._nodes[claim.id] = data
        if HAS_NX:
            self.G.add_node(claim.id, **data)

    def add_assumption_node(self, assumption: Assumption):
        data = {
            "type": "assumption", "constraint": assumption.constraint,
            "explicit": assumption.explicit,
            "verification": str(assumption.verification),
            "score": assumption.score,
        }
        self._nodes[assumption.id] = data
        if HAS_NX:
            self.G.add_node(assumption.id, **data)

    def add_agreement_edge(self, agreement: Agreement):
        edge = {
            "source": agreement.claim_i_id, "target": agreement.claim_j_id,
            "relation": agreement.relation, "confidence": agreement.confidence,
            "reason": agreement.reason,
        }
        self._edges.append(edge)
        if HAS_NX:
            self.G.add_edge(
                agreement.claim_i_id, agreement.claim_j_id,
                **{k: v for k, v in edge.items() if k not in ("source","target")},
            )

    def add_assumption_link(self, claim_id: str, assumption_id: str):
        edge = {"source": claim_id, "target": assumption_id, "relation": "depends_on"}
        self._edges.append(edge)
        if HAS_NX:
            self.G.add_edge(claim_id, assumption_id, relation="depends_on")

    # ── A: COMMUNITY DETECTION ────────────────────────────────────────────────

    def detect_communities(self) -> Dict[str, Any]:
        """
        Greedy modularity community detection on the claim subgraph.
        Returns {
            "clusters": [[node_id, ...], ...],   # each cluster is a list of claim IDs
            "membership": {node_id: cluster_idx}, # for fast lookup
            "theme_domains": ["NLP", "CV", ...]  # dominant domain per cluster
        }
        """
        if not HAS_NX or self.G.number_of_nodes() < 2:
            return {"clusters": [], "membership": {}, "theme_domains": []}

        claim_nodes = {n for n, d in self._nodes.items() if d.get("type") == "claim"}
        if len(claim_nodes) < 2:
            return {"clusters": [list(claim_nodes)], "membership": {n:0 for n in claim_nodes}, "theme_domains": []}

        subG = self.G.subgraph(claim_nodes).to_undirected()
        # Add missing nodes (isolated claims)
        for n in claim_nodes:
            if n not in subG:
                subG = nx.Graph(subG)
                subG.add_node(n)

        try:
            raw_comms = list(nx_community.greedy_modularity_communities(subG))
        except Exception as e:
            logger.debug("community detection: %s", e)
            return {"clusters": [list(claim_nodes)], "membership": {n:0 for n in claim_nodes}, "theme_domains": []}

        clusters = [sorted(list(c)) for c in raw_comms]
        membership = {}
        for idx, cluster in enumerate(clusters):
            for nid in cluster:
                membership[nid] = idx

        # Infer theme per cluster from dominant domain
        theme_domains = []
        for cluster in clusters:
            domains = [self._nodes.get(n, {}).get("domain", "") for n in cluster if self._nodes.get(n, {}).get("domain")]
            if domains:
                dominant = max(set(domains), key=domains.count)
                theme_domains.append(dominant)
            else:
                theme_domains.append(f"Cluster {len(theme_domains)+1}")

        logger.info("EDG communities: %d clusters from %d claims", len(clusters), len(claim_nodes))
        return {"clusters": clusters, "membership": membership, "theme_domains": theme_domains}

    # ── B: INFLUENCE PROPAGATION ──────────────────────────────────────────────

    def influence_propagation(self, iterations: int = 3) -> Dict[str, float]:
        """
        Weighted iterative uncertainty propagation.

        Update rule:
            u[n] = 0.7 * u[n]  +  0.3 * avg(w_e * u[neighbour])
        Edge weights:
            contradict → 1.4  (amplifies uncertainty)
            agree      → 0.8  (dampens it)
            conditional→ 1.1  (mild amplification)
            depends_on → 0.5  (assumption links, weak bleed)

        Returns {node_id: propagated_uncertainty}
        """
        if not HAS_NX:
            return {}

        claim_nodes = {n for n, d in self._nodes.items() if d.get("type") == "claim"}
        u = {n: self._nodes[n].get("uncertainty", 0.3) for n in claim_nodes}

        edge_weights = {"contradict": 1.4, "agree": 0.8, "conditional": 1.1, "depends_on": 0.5}
        undirected = self.G.to_undirected()

        for _ in range(iterations):
            new_u = {}
            for n in claim_nodes:
                nbrs = [v for v in undirected.neighbors(n) if v in claim_nodes]
                if not nbrs:
                    new_u[n] = u[n]
                    continue
                weighted = []
                for nb in nbrs:
                    edge_data = undirected.get_edge_data(n, nb) or {}
                    rel = edge_data.get("relation", "agree")
                    w   = edge_weights.get(rel, 0.8)
                    weighted.append(u[nb] * w)
                avg_weighted = sum(weighted) / len(weighted)
                new_u[n] = round(min(u[n] * 0.7 + avg_weighted * 0.3, 1.0), 3)
            u = new_u

        return u

    # ── C: REASONING PATHS ───────────────────────────────────────────────────

    def reasoning_paths(self) -> List[Dict]:
        """
        For every contradiction pair, find the shortest reasoning path.
        Returns list of:
        {
            "from": "C1", "to": "C4",
            "path": ["C1","A1","C3","C4"],
            "path_text": ["LLM outperforms", "assumes GPU", "..."],
            "interpretation": "DIRECT_CONTRADICTION | INDIRECT_CONTRADICTION | TRANSITIVE_SUPPORT"
        }
        """
        if not HAS_NX:
            return []

        undirected = self.G.to_undirected()
        claim_nodes = {n for n, d in self._nodes.items() if d.get("type") == "claim"}
        contra_pairs = [
            (e["source"], e["target"])
            for e in self._edges
            if e.get("relation") == "contradict"
            and e["source"] in claim_nodes
            and e["target"] in claim_nodes
        ]

        paths = []
        seen = set()
        for src, tgt in contra_pairs:
            key = tuple(sorted([src, tgt]))
            if key in seen:
                continue
            seen.add(key)

            try:
                path = nx.shortest_path(undirected, source=src, target=tgt)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = [src, tgt]

            # Get text labels for each node in path
            path_text = []
            for nid in path:
                d = self._nodes.get(nid, {})
                if d.get("type") == "claim":
                    path_text.append(d.get("text", nid)[:40])
                elif d.get("type") == "assumption":
                    path_text.append(f"[A] {d.get('constraint', nid)[:30]}")
                else:
                    path_text.append(nid[:20])

            # Classify the path
            edges_on_path = []
            for i in range(len(path) - 1):
                ed = undirected.get_edge_data(path[i], path[i+1]) or {}
                edges_on_path.append(ed.get("relation", "unknown"))

            if len(path) == 2:
                interp = "DIRECT_CONTRADICTION"
            elif any(r == "contradict" for r in edges_on_path):
                interp = "INDIRECT_CONTRADICTION"
            elif all(r == "agree" for r in edges_on_path):
                interp = "TRANSITIVE_SUPPORT"
            else:
                interp = "MIXED_PATH"

            paths.append({
                "from": src, "to": tgt,
                "path": path,
                "path_text": path_text,
                "edge_types": edges_on_path,
                "interpretation": interp,
                "length": len(path),
            })

        # Also add paths for high-uncertainty → gap region chains
        gap_regs = self.gap_regions()
        for gap_nid in gap_regs[:3]:
            # Find the highest-certainty claim and trace path to this gap
            claim_u = {n: self._nodes[n].get("uncertainty", 0) for n in claim_nodes}
            anchor = min(claim_u, key=claim_u.get, default=None)
            if anchor and anchor != gap_nid:
                try:
                    path = nx.shortest_path(undirected, source=anchor, target=gap_nid)
                    if len(path) >= 2:
                        path_text = []
                        for nid in path:
                            d = self._nodes.get(nid, {})
                            if d.get("type") == "claim":
                                path_text.append(d.get("text", nid)[:40])
                            elif d.get("type") == "assumption":
                                path_text.append(f"[A] {d.get('constraint', nid)[:30]}")
                            else:
                                path_text.append(nid[:20])
                        paths.append({
                            "from": anchor, "to": gap_nid,
                            "path": path, "path_text": path_text,
                            "edge_types": [],
                            "interpretation": "ANCHOR_TO_GAP",
                            "length": len(path),
                        })
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass

        return paths

    # ── Standard metrics ──────────────────────────────────────────────────────

    def betweenness_centrality(self) -> Dict[str, float]:
        if not HAS_NX or self.G.number_of_nodes() < 3:
            return {}
        try:
            return nx.betweenness_centrality(self.G.to_undirected(), normalized=True)
        except Exception as e:
            logger.debug("betweenness: %s", e); return {}

    def pagerank(self) -> Dict[str, float]:
        if not HAS_NX or self.G.number_of_nodes() < 2:
            return {}
        try:
            claim_nodes = {n for n, d in self._nodes.items() if d.get("type") == "claim"}
            subG = self.G.subgraph(claim_nodes)
            return nx.pagerank(subG, alpha=0.85) if subG.number_of_edges() > 0 else {}
        except Exception as e:
            logger.debug("pagerank: %s", e); return {}

    def clustering_coefficient(self) -> Dict[str, float]:
        if not HAS_NX or self.G.number_of_nodes() < 3:
            return {}
        try:
            return nx.clustering(self.G.to_undirected())
        except Exception as e:
            logger.debug("clustering: %s", e); return {}

    def contradiction_clusters(self) -> List[List[str]]:
        if not HAS_NX:
            return [[e["source"],e["target"]] for e in self._edges if e.get("relation")=="contradict"]
        cg = nx.Graph()
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "contradict":
                cg.add_edge(u, v)
        return [list(c) for c in nx.connected_components(cg)]

    def gap_regions(self, degree_threshold: int = 2, bc_threshold: float = 0.05) -> List[str]:
        claim_nodes = {n for n, d in self._nodes.items() if d.get("type") == "claim"}
        if not claim_nodes:
            return []
        bc = self.betweenness_centrality()
        low_bc  = {n for n in claim_nodes if bc.get(n, 0) < bc_threshold}
        if HAS_NX:
            ug = self.G.to_undirected()
            low_deg = {n for n in claim_nodes if ug.degree(n) < degree_threshold}
        else:
            edge_counts = {n: 0 for n in claim_nodes}
            for e in self._edges:
                if e.get("relation") != "depends_on":
                    for s in ("source","target"):
                        if e[s] in edge_counts:
                            edge_counts[e[s]] += 1
            low_deg = {n for n, c in edge_counts.items() if c < degree_threshold}
        return list(low_deg | low_bc)

    def average_uncertainty(self) -> float:
        u = [d["uncertainty"] for d in self._nodes.values()
             if d.get("type")=="claim" and "uncertainty" in d]
        return round(sum(u)/max(len(u),1), 3)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        bc      = self.betweenness_centrality()
        pr      = self.pagerank()
        cl      = self.clustering_coefficient()
        g_regs  = self.gap_regions()
        inf_u   = self.influence_propagation()   # B
        comms   = self.detect_communities()       # A
        rpaths  = self.reasoning_paths()          # C
        cp      = next(
            (p["path"] for p in rpaths if p["interpretation"] == "DIRECT_CONTRADICTION"),
            None
        )

        nodes_out = []
        for k, v in self._nodes.items():
            n = {"id": k, **v}
            if k in bc:     n["betweenness"]           = round(bc[k], 4)
            if k in pr:     n["pagerank"]               = round(pr[k], 4)
            if k in cl:     n["clustering"]             = round(cl[k], 4)
            if k in inf_u:  n["influence_uncertainty"]  = inf_u[k]
            if k in comms["membership"]:
                n["community"]    = comms["membership"][k]
                n["community_theme"] = comms["theme_domains"][comms["membership"][k]] \
                    if comms["membership"][k] < len(comms["theme_domains"]) else ""
            if k in g_regs: n["gap_region"] = True
            nodes_out.append(n)

        return {
            "nodes": nodes_out,
            "edges": self._edges,
            "analytics": {
                "contradiction_path": cp,
                "reasoning_paths":    rpaths,
                "communities":        comms["clusters"],
                "community_themes":   comms["theme_domains"],
                "num_communities":    len(comms["clusters"]),
                "num_gap_regions":    len(g_regs),
                "contra_clusters":    len(self.contradiction_clusters()),
            },
            "stats": {
                "num_claims":          sum(1 for v in self._nodes.values() if v["type"]=="claim"),
                "num_assumptions":     sum(1 for v in self._nodes.values() if v["type"]=="assumption"),
                "num_edges":           len(self._edges),
                "avg_uncertainty":     self.average_uncertainty(),
                "contradiction_count": sum(1 for e in self._edges if e.get("relation")=="contradict"),
                "gap_region_count":    len(g_regs),
                "contra_clusters":     len(self.contradiction_clusters()),
            },
        }


def build_edg(claims: List[Claim], agreements: List[Agreement]) -> EpistemicDependencyGraph:
    edg = EpistemicDependencyGraph()
    for claim in claims:
        edg.add_claim_node(claim)
        for assumption in claim.assumptions:
            edg.add_assumption_node(assumption)
            edg.add_assumption_link(claim.id, assumption.id)
    for ag in agreements:
        edg.add_agreement_edge(ag)
    logger.info("EDG: %d nodes, %d edges.", len(edg._nodes), len(edg._edges))
    return edg
