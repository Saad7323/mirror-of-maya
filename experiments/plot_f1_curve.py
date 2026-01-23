import matplotlib.pyplot as plt
from src.pipeline import DuplicateDetector
from src.evaluation.dataset_loader import ImagePairDataset
from src.similarity.thresholding import apply_threshold
from src.evaluation.metrics import compute_metrics
import numpy as np

def main():
    dataset = ImagePairDataset("data/pairs.csv", "data/raw")

    detector = DuplicateDetector(
        phash_threshold=18,
        clip_threshold=0.0  # dummy
    )

    scores = []
    labels = []

    for sample in dataset:
        score, _ = detector.compare(sample["img1"], sample["img2"])
        scores.append(score)
        labels.append(sample["label"])

    thresholds = np.arange(0.3, 0.9, 0.02)
    f1_scores = []

    for t in thresholds:
        preds = [apply_threshold(s, t) for s in scores]
        _, _, f1 = compute_metrics(labels, preds)
        f1_scores.append(f1)

    plt.plot(thresholds, f1_scores, marker="o")
    plt.xlabel("CLIP Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs CLIP Threshold (Hybrid Model)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
