from PIL import image
import imagehash

def compute_phash(image_path:str): #computes phash of image to a 64 bit perceptual hash
    image = Image.open(image_path).convert("RGB") 
    phash = imagehash.phash(image) #computes perceptual hash using DCT-based method

    return phash
