import torch
import clip
import numpy as np
from PIL import Image

#deciding whether to use CPU or GPU 

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model 

model, preprocess = clip.load("ViT-B/32", device=device)  #Loads the vision transformer and outputs a 512-dimensional vector that represents what is in the image(not pixels but meaning)
model.eval() #set model to evaluation mode

def get_embedding(image_path: str) -> np.ndarray:  #converts image into normalized clip embedding, returns a 512-d normalized vector representation of image
    image = Image.open(image_path).convert("RGB") #Load image from disk and convert it to RGB format

    #Apply CLIP's preprocessing: Resize, Center crop, Normalize pixel values
    image_tensor = preprocess(image).unsqueeze(0).to(device) #Adding a batch dimension using unsqueeze(0) then move tensor to CPU or GPU

    #Disabling gradient computation for faster inference as we are not training the model
    with torch.no_grad():
        embedding = model.encode_image(image_tensor) #Passing image tensor through CLIP's image encoder to produce a 512-d feature vector
    
    #Normalize embedding to unit length to ensure cosine similarity works
    embedding = embedding / embedding.norm(dim=-1, keepdim=True) #After this step, dot product == cosine similarity

    return embedding.squeeze(0).cpu().numpy() #Remove batch dimension using squeeze(0) then moves the tensor to memory converting it into NumPy array for similarity computation