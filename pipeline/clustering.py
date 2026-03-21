# pipeline/clustering.py

import numpy as np
from typing import Any
from sklearn.cluster import AgglomerativeClustering
from pipeline.embedding import EmbeddingEngine

MIN_PAPERS_TO_CLUSTER = 3   # don't bother clustering fewer than this
MAX_CLUSTERS = 6            # cap to avoid over-fragmentation
DISTANCE_THRESHOLD = 0.35   # cosine distance threshold (1 - similarity)
                            # 0.35 → papers with similarity >= 0.65 group together


class PaperClusterer:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        distance_threshold: float = DISTANCE_THRESHOLD,
        max_clusters: int = MAX_CLUSTERS,
        debug: bool = True,
    ):
        self.engine = embedding_engine
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.debug = debug

    def _get_embeddings(self, papers: list[dict[str, Any]]) -> tuple[list[int], np.ndarray]:
        # only cluster papers that have embeddings
        valid_indices = []
        embeddings = []

        for i, paper in enumerate(papers):
            emb = paper.get("embedding")
            if emb is not None:
                valid_indices.append(i)
                embeddings.append(emb)

        return valid_indices, np.array(embeddings) if embeddings else np.array([])

    def _label_cluster(self, papers: list[dict[str, Any]]) -> str:
        # use the title of the highest-scored paper in the cluster as label
        sorted_papers = sorted(papers, key=lambda p: p.get("score", 0.0), reverse=True)
        title = sorted_papers[0].get("title", "Unknown")
        # truncate long titles
        return title if len(title) <= 60 else title[:57] + "..."

    def cluster(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(papers) < MIN_PAPERS_TO_CLUSTER:
            if self.debug:
                print(f"[Clusterer] Too few papers ({len(papers)}), skipping clustering")
            for paper in papers:
                paper["cluster_id"] = 0
                paper["cluster_label"] = "general"
            return papers

        valid_indices, embeddings = self._get_embeddings(papers)

        if len(valid_indices) < MIN_PAPERS_TO_CLUSTER:
            if self.debug:
                print("[Clusterer] Not enough papers with embeddings")
            for paper in papers:
                paper["cluster_id"] = 0
                paper["cluster_label"] = "general"
            return papers

        # agglomerative clustering with cosine distance
        n_papers = len(valid_indices)
        model = AgglomerativeClustering(
            metric="cosine",
            linkage="average",          # average linkage works well with cosine
            distance_threshold=self.distance_threshold,
            n_clusters=None,            # let threshold decide number of clusters
        )

        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels))

        # if too many clusters, re-run with relaxed threshold
        if n_clusters > self.max_clusters:
            if self.debug:
                print(f"[Clusterer] {n_clusters} clusters exceeds max {self.max_clusters}, relaxing threshold")
            relaxed_threshold = self.distance_threshold + 0.1
            model = AgglomerativeClustering(
                metric="cosine",
                linkage="average",
                distance_threshold=relaxed_threshold,
                n_clusters=None,
            )
            labels = model.fit_predict(embeddings)
            n_clusters = len(set(labels))

        if self.debug:
            print(f"[Clusterer] Formed {n_clusters} clusters from {n_papers} papers")

        # attach cluster_id to valid papers
        for idx, label in zip(valid_indices, labels):
            papers[idx]["cluster_id"] = int(label)

        # papers without embeddings get cluster -1
        for i, paper in enumerate(papers):
            if i not in valid_indices:
                paper["cluster_id"] = -1

        # build cluster groups for labeling
        clusters: dict[int, list[dict]] = {}
        for paper in papers:
            cid = paper.get("cluster_id", -1)
            clusters.setdefault(cid, []).append(paper)

        # generate label for each cluster
        cluster_labels: dict[int, str] = {}
        for cid, cluster_papers in clusters.items():
            if cid == -1:
                cluster_labels[cid] = "unclustered"
            else:
                cluster_labels[cid] = self._label_cluster(cluster_papers)

        # attach labels
        for paper in papers:
            cid = paper.get("cluster_id", -1)
            paper["cluster_label"] = cluster_labels.get(cid, "unknown")

        if self.debug:
            for cid, label in cluster_labels.items():
                count = len(clusters[cid])
                print(f"[Clusterer] Cluster {cid} ({count} papers): {label}")

        return papers