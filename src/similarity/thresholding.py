import numpy as np
from sklearn.metrics import f1_score


def apply_threshold(similarity_score: float, threshold: float) -> int:

#Converts a similarity score to a binary decision (0 or 1).

    if similarity_score >= threshold:
        return 1
    else:
        return 0


def find_optimal_threshold(similarity_scores, true_labels, step=0.01):
    """
    Finds the optimal threshold that maximizes F1 score on the given data.
    
    Args:
        similarity_scores: List or array of similarity scores (float values between 0.0 and 1.0)
        true_labels: List or array of true labels (0 or 1), where 1 = duplicate, 0 = not duplicate
        step: How much to increment the threshold each time (default 0.01 means try 0.0, 0.01, 0.02, ...)
    
    Returns:
        The optimal threshold (float) that maximizes F1 score
    """
    similarity_scores = np.array(similarity_scores)
    true_labels = np.array(true_labels)
    
    best_threshold = 0.0
    best_f1 = 0.0
    
    thresholds_to_try = np.arange(0.3, 0.9 + step, step)#0.5 to 0.95 as thats where the cosine similarity values become meaningful for CLIP embeddings.
    
    for threshold in thresholds_to_try: # Apply this threshold to all similarity scores
        
        predictions = [apply_threshold(score, threshold) for score in similarity_scores]
        
        f1 = f1_score(true_labels, predictions,zero_division=0) #zero_division to prevent f1 score from crashing when there is extreme threshold
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold,best_f1
