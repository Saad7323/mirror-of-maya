from src.pipeline import DuplicateDetector
from src.evaluation.dataset_loader import ImagePairDataset
from src.similarity.thresholding import apply_threshold
import csv


def main():
    detector = DuplicateDetector(
        mode="hybrid",
        phash_threshold=18,
        clip_threshold=0.30
    )

    dataset = ImagePairDataset("data/pairs.csv", "data/raw")

    false_negatives = []

    for sample in dataset:
        score, decision = detector.compare(
            sample["img1"],
            sample["img2"]
        )

        # False Negative: true label = 1, predicted = 0
        if sample["label"] == 1 and not decision:
            false_negatives.append([
                sample["img1"],
                sample["img2"],
                score
            ])

    print(f"False Negatives found: {len(false_negatives)}")

    with open("results/false_negatives.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_1", "image_2", "similarity_score"])
        writer.writerows(false_negatives)

    print("Saved to results/false_negatives.csv")


if __name__ == "__main__":
    main()
