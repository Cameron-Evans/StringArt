import numpy as np
from PIL import Image, ImageDraw
import sys

def main():
    # Open the input image as numpy array, convert to RGB
    img=Image.open("images/Orca.png").convert("RGB")
    img = cropToCircle(img)

    img.save('result.png')


# returns a cropped version the given PIL.Image so it is circular/ovular
def cropToCircle(image):
    # Open the input image as numpy array, convert to RGB
    npImage=np.array(image)
    h,w=image.size

    # Create same size alpha layer with circle
    alpha = Image.new('L', (h,w), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0,0,h,w],0,360,fill=255)

    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)

    # Add alpha layer to RGB
    npImage=np.dstack((npImage,npAlpha))

    return Image.fromarray(npImage)




if __name__ == "__main__":
    main()