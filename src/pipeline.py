from embedding.clip_encoder import get_embedding
from similarity.similarity_metrics import cosine_similarity

threshold = 0.8

def main():
    img1_path = r"C:\Users\rkesh\Downloads\test1.jpg"
    img2_path = r"C:\Users\rkesh\Downloads\test2.jpg"
    
    #Embedding
    emb1 = get_embedding(img1_path)
    emb2 = get_embedding(img2_path)
    print("Embedding Shapes: ", emb1.shape,emb2.shape)

    #Similarity
    score = cosine_similarity(emb1,emb2)
    print("Cosine Similarity: {score:.4f}")

    #Decision
    if score >= threshold:
        print("NEAR DUPLICATE")
    else:
        print("NOT DUPLICATE")

if __name__ == "__main__":
    main()