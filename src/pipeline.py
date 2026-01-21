from embedding.clip_encoder import get_embedding
from embedding.phash_encoder import get_phash
from similarity.similarity_metrics import cosine_similarity

phash_threshold = 15        # Hamming distance
clip_threshold = 0.8       #cosine similarity

def test_pair(img1_path, img2_path):
    #img1_path = "data/raw/original/img1.jpg"
    #img2_path = "data/raw/modified/img1_crop.jpg"

    phash1 = get_phash(img1_path)
    phash2 = get_phash(img2_path)

    hamming_distance = phash1 - phash2
    print(f"pHash Hamming Distance: {hamming_distance}")

    if hamming_distance > phash_threshold:
        print("NOT DUPLICATE (Rejected by pHash)")
        return

    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    print("Embedding Shapes:", emb1.shape, emb2.shape)

    score = cosine_similarity(emb1, emb2)
    print(f"Cosine Similarity: {score:.4f}")

    if score >= clip_threshold:
        print("NEAR DUPLICATE")
    else:
        print("NOT DUPLICATE")

def main():
    test_pair(
        "data/raw/original/img14.jpg",
        "data/raw/modified/img1.jpg"
    )

    test_pair(
        "data/raw/original/img24.jpg",
        "data/raw/original/img32.jpg"
    )

if __name__ == "__main__":
    main()
