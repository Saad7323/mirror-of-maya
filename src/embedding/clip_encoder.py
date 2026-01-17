import torch
import clip
import numpy as np
from PIL import Image

#deciding whether to use CPU or GPU 

device = "cuda" if torch.cude.is_available() else "cpu"