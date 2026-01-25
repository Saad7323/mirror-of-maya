# Mirror of Maya
**Near-Duplicate Image Detection using CLIP and Perceptual Hashing**

This project focuses on detecting near-duplicate and semantically similar images using a combination of deep learning–based image embeddings and perceptual hashing techniques. 
The system is designed to identify visually or conceptually similar images even under transformations such as resizing, cropping, compression, or minor edits.

**Overview**
Traditional image deduplication techniques rely on exact pixel matching or simple hashing methods, which fail when images undergo slight visual or semantic changes.
This project adopts a modern hybrid approach, leveraging CLIP embeddings for semantic similarity and perceptual hashing (pHash) for fast visual filtering, enabling robust near-duplicate detection.

**Methodology**

Dataset Preparation:
  Used an image dataset containing both duplicate / near-duplicate pairs and non-matching image pairs
  Generated positive and negative image pairs for evaluation
  Ensured consistent preprocessing (resizing, normalization) before feature extraction

**Model Architecture**

CLIP-based Similarity Model
  Image encoder from CLIP generates high-dimensional embeddings
  Cosine similarity used to measure semantic similarity between image pairs

Hybrid Pipeline (pHash + CLIP)
  pHash used as a fast pre-filter to eliminate obvious non-matches
  CLIP similarity applied only to shortlisted candidates
  Thresholding used to convert similarity scores into match / no-match decisions

**Training**

No supervised training required (pretrained CLIP model). Performed batch-wise inference using PyTorch.
Implemented deterministic evaluation where possible. Threshold tuning conducted to study performance variation.

**Evaluation Metrics**

The system is evaluated using standard classification metrics:
  Precision – correctness of predicted matches
  Recall – ability to detect true duplicates
  F1 Score – balance between precision and recall

Additional analysis:
  F1 score vs CLIP threshold curves
  Comparison between CLIP-only and Hybrid (pHash + CLIP) approaches
  Stability analysis across multiple runs

  
