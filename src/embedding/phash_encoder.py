from PIL import Image
import imagehash 

def get_phash(image_path:str): #computes phash of image to a 64 bit perceptual hash
    image = Image.open(image_path).convert("RGB") 
    phash = imagehash.phash(image) #computes perceptual hash using DCT-based method

    return phash

#phash is perceptual hash it turns images into small fingerprints that represents how it looks, not how its pixels are stored.