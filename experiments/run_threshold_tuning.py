from src.pipeline import DuplicateDetector
from src.evaluation.dataset_loader import ImagePairDataset
from src.similarity.thresholding import find_optimal_threshold, apply_threshold
from src.evaluation.metrics import compute_metrics


def main():
    # Initialize detector (hybrid: pHash + CLIP)
    detector = DuplicateDetector(
        phash_threshold=18,   # keep pHash fixed
        clip_threshold=0.3    # dummy initial value (will be tuned)
    )

    # Load dataset
    dataset = ImagePairDataset(
        csv_path="data/pairs.csv",
        root_dir="data/raw"
    )

    similarity_scores = []
    true_labels = []

    print("Running inference on dataset...")

    for sample in dataset:
        score, decision = detector.compare(
            sample["img1"],
            sample["img2"]
        )

        similarity_scores.append(score)
        true_labels.append(sample["label"])

    print("Done.")
    print(f"Total pairs evaluated: {len(similarity_scores)}")

    # Threshold tuning
    best_threshold, best_f1 = find_optimal_threshold(
        similarity_scores,
        true_labels
    )

    print("\nOptimal Threshold Found")
    print("-----------------------")
    print(f"Best CLIP Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_f1:.4f}")

    # Final evaluation using best threshold
    predictions = [
        apply_threshold(score, best_threshold)
        for score in similarity_scores
    ]

    precision, recall, f1 = compute_metrics(true_labels, predictions)

    print("\nFinal Evaluation Metrics")
    print("------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    main()
