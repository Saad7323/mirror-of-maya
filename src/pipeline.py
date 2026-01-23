from src.embedding.clip_encoder import get_embedding
from src.embedding.phash_encoder import get_phash
from src.similarity.similarity_metrics import cosine_similarity


class DuplicateDetector:
    def __init__(self, phash_threshold=15, clip_threshold=0.8):
        self.phash_threshold = phash_threshold
        self.clip_threshold = clip_threshold

    def compare(self, img1_path, img2_path):
        # Step 1: p-Hash Filtering
        phash1 = get_phash(img1_path)
        phash2 = get_phash(img2_path)

        hamming_distance = phash1 - phash2

        if hamming_distance > self.phash_threshold:
            return 0.0, False  # Rejected Early

        # Step 2: CLIP Similarity
        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)

        score = cosine_similarity(emb1, emb2)
        decision = score >= self.clip_threshold

        return score, decision
