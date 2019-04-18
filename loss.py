import numpy as np
from math import sqrt

def compare(image1, image2):
    return (sqrt((image1-image2)**2))
