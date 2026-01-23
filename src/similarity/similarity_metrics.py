import numpy as np 

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float: #to hint that vec1 and vec2 are numpy arrays and float is to determine that the return value will be a float value.
    dot_product = np.dot(vec1,vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0.0 or norm_vec2 == 0.0:
        return 0.0
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return float(similarity)

#so cosine similarity is basically just to check how aligned the photos are, and the more the value of it
#the more duplicate it is. As image embeddings are already vectors it will make it easier to compare between images.
#NOTE: Cosine similarity is not similarity of size, its similarity of direction.