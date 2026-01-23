import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import DuplicateDetector
from src.evaluation.dataset_loader import ImagePairDataset
from src.similarity.thresholding import apply_threshold
from src.evaluation.metrics import compute_metrics


def compute_f1_curve(mode, thresholds, phash_threshold=18):
    dataset = ImagePairDataset("data/pairs.csv", "data/raw")

    detector = DuplicateDetector(
        mode=mode,
        phash_threshold=phash_threshold,
        clip_threshold=0.0  # dummy, overridden by sweep
    )

    scores = []
    labels = []

    for sample in dataset:
        score, _ = detector.compare(sample["img1"], sample["img2"])
        scores.append(score)
        labels.append(sample["label"])

    f1_scores = []
    for t in thresholds:
        preds = [apply_threshold(s, t) for s in scores]
        _, _, f1 = compute_metrics(labels, preds)
        f1_scores.append(f1)

    return f1_scores


def main():
    thresholds = np.arange(0.3, 0.9, 0.02)

    # Hybrid
    hybrid_f1 = compute_f1_curve(
        mode="hybrid",
        thresholds=thresholds
    )

    # CLIP-only
    clip_f1 = compute_f1_curve(
        mode="clip",
        thresholds=thresholds
    )

    #Hybrid
    plt.figure()
    plt.plot(thresholds, hybrid_f1, marker="o")
    plt.xlabel("CLIP Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs CLIP Threshold (Hybrid Model)")
    plt.grid(True)
    plt.savefig("results/f1_vs_threshold_hybrid.png")
    plt.close()

    #Comparison
    plt.figure()
    plt.plot(thresholds, clip_f1, label="CLIP-only", marker="o")
    plt.plot(thresholds, hybrid_f1, label="Hybrid (pHash + CLIP)", marker="o")
    plt.xlabel("CLIP Threshold")
    plt.ylabel("F1 Score")
    plt.title("CLIP-only vs Hybrid (F1 Comparison)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/clip_vs_hybrid_f1.png")
    plt.close()

    print("Plots saved in results/")


if __name__ == "__main__":
    main()
