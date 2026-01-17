import csv
import os


class ImagePairDataset:
    def __init__(self, csv_path, root_dir):
        """
        csv_path: path to pairs.csv
        root_dir: path to data/raw
        """
        self.root_dir = root_dir
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "img1": os.path.join(root_dir, row["image_1"]),
                    "img2": os.path.join(root_dir, row["image_2"]),
                    "label": int(row["label"])
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
