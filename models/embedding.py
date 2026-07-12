import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from pathlib import Path
from typing import List, Optional

from configs.settings import configure_logger
from configs.paths import EMBEDDINGS_DIR

logger = configure_logger("EmbeddingProjector")

class EmbeddingProjector:
    def __init__(self, embeddings: np.ndarray, labels: List[int], class_names: List[str]):
        """
        Handles dimension reduction projections (t-SNE and UMAP) for 512-dimensional embeddings.
        """
        self.embeddings = embeddings
        self.labels = np.array(labels)
        self.class_names = class_names

    def project_tsne(self, perplexity: int = 30, max_iter: int = 1000) -> np.ndarray:
        """Projects high-dim embeddings into 2D space using t-SNE."""
        logger.info("Computing t-SNE 2D projections...")
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
        projections = tsne.fit_transform(self.embeddings)
        return projections

    def project_umap(self, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """Projects high-dim embeddings into 2D space using UMAP."""
        logger.info("Computing UMAP 2D projections...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        projections = reducer.fit_transform(self.embeddings)
        return projections

    def plot_projection(self, projections: np.ndarray, title: str, save_path: Optional[Path] = None) -> None:
        """Plots scatter visualization colored by categories."""
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(self.labels)
        palette = sns.color_palette("tab10", len(unique_labels))
        
        for idx, label in enumerate(unique_labels):
            mask = self.labels == label
            plt.scatter(
                projections[mask, 0], projections[mask, 1],
                label=self.class_names[label], alpha=0.7, edgecolors='none', s=20, color=palette[idx]
            )
            
        plt.title(title, fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Categories")
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved embedding projection plot to: {save_path}")
        plt.close()
