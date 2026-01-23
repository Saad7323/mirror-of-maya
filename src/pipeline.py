from src.embedding.clip_encoder import get_embedding
from src.embedding.phash_encoder import get_phash
from src.similarity.similarity_metrics import cosine_similarity


class DuplicateDetector:
    def __init__(self, mode="hybrid", phash_threshold=18, clip_threshold=0.30):
        self.mode = mode
        self.phash_threshold = phash_threshold
        self.clip_threshold = clip_threshold

    def compare(self, img1_path, img2_path):
        #pHash 
        if self.mode == "phash":
            phash1 = get_phash(img1_path)
            phash2 = get_phash(img2_path)
            distance = phash1 - phash2 #hamming distance
            decision = distance <= self.phash_threshold
            return float(distance), decision

        #CLIP
        if self.mode == "clip":
            emb1 = get_embedding(img1_path)
            emb2 = get_embedding(img2_path)
            score = cosine_similarity(emb1, emb2)
            decision = score >= self.clip_threshold
            return score, decision

        #Hybrd
        phash1 = get_phash(img1_path)
        phash2 = get_phash(img2_path)
        distance = phash1 - phash2 #hamming distance

        if distance > self.phash_threshold:
            return 0.0, False #rejected early

        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)
        score = cosine_similarity(emb1, emb2)
        decision = score >= self.clip_threshold
        return score, decision

