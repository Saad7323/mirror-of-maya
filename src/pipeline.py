from src.embedding.clip_encoder import get_embedding
from src.similarity.similarity_metrics import cosine_similarity

threshold = 0.8

def main():
    img1_path = "data/raw/original/img1.jpg"
    img2_path = "data/raw/modified/img1_crop.jpg"

    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)

    print("Embedding Shapes:", emb1.shape, emb2.shape)

    score = cosine_similarity(emb1, emb2)
    print(f"Cosine Similarity: {score:.4f}")

    if score >= threshold:
        print("NEAR DUPLICATE")
    else:
        print("NOT DUPLICATE")

if __name__ == "__main__":
    main()
