from asyncore import file_dispatcher
from cmath import pi
from PIL import Image, ImageDraw
from bresenham import bresenham
import numpy as np
import sys
import math
import random

#Parameters
numOfEdgeNails = 0 #helps determine size of piece. argv[1]
numOfFieldNails = 0 #adds complexity and detail. argv[2]
minDistanceBetweenNails = 0 #measured in mm, helps determine size of piece. argv[3]
ThreadLen = 0 #measured in meters. split among CYMk argv[4]
neighborExclusion = 0 #how many neighbors must a connection between edgeNails skip. Encourages threads to pass through the middle. argv[5]
minDistanceFNailFromEdge = 0 #minimum distance in mm a field nail can lie from the edge. argv[6]
proportionOfThreadInk = 1.0 #proportion of color removed from .png per thread. argv[7]

#Globals (not handed to program, but determined by parameters)
edgeNails = [(0,0)] #Series of XY coordinates, specifying the location of each edge nail
fieldNails = [(0,0)] #Series of XY coordinates, specifying the location of each field nail
Cpath = []
Mpath = []
Ypath = []
Kpath = []


#The input photo will be resized so that each pixel is one square millimeter
#It is assumed that the thread is one millimeter (one pixel) thick


def main():
    if (len(sys.argv) != 8):
        print("Correct usage: python3 main.py [numberOfEdgeNails] [numberOfFieldNails] [minimumDistanceBetweenNails] [metersOfColoredThread] [neighbor exclusion] [field nail distance from edge] [prop of thread ink]")
        exit()

    #Fill global parameters with cmd line args
    globals()['numOfEdgeNails'] = int(sys.argv[1])
    globals()['numOfFieldNails'] = int(sys.argv[2])
    globals()['minDistanceBetweenNails'] = int(sys.argv[3])
    globals()['ThreadLen'] = int(sys.argv[4])
    globals()['neighborExclusion'] = int(sys.argv[5])
    globals()['minDistanceFNailFromEdge'] = int(sys.argv[6])
    globals()['proportionOfThreadInk'] = float(sys.argv[7])

    print("With the provided parameters, final piece will be:", numOfEdgeNails*minDistanceBetweenNails/math.pi/1000, "meters across")

    # Open the input image as numpy array, convert to RGB
    img=Image.open("images/Snow.png").convert("RGB")

    # Resize the image to a square of width based on parameters
    newsize = round(numOfEdgeNails*minDistanceBetweenNails/math.pi)
    if (newsize%2 == 0): #odd edge length is needed
        newsize =+ 1
    newsize = (newsize, newsize)
    img = img.resize(newsize)

    img = cropToCircle(img)
    print("Output image will be:",img.size, "mm")

    #populate globals
    print("Placing",numOfEdgeNails,"edge nails")
    placeEdgeNails(newsize[0]//2)


    print("Placing",numOfFieldNails,"field nails")
    placeFieldNails(img)
    print("Field nail coordinates:")
    for i,xy in enumerate(fieldNails):
        print(xy, "as field nail number:", i)

    print("Allocating",ThreadLen,"meters of thread")
    lengths = determineLengths(img)

    print("Generating cyan thread path with",lengths[0]*ThreadLen,"meters of thread")
    globals()['Cpath'] = generateThreadPath('C', lengths[0]*ThreadLen*1000, img)
    print(len(Cpath), "cyan connections made")
    print("Generating magenta thread path with" , lengths[1]*ThreadLen , "meters of thread")
    globals()['Mpath'] = generateThreadPath('M', lengths[1]*ThreadLen*1000, img)
    print(len(Mpath), "magenta connections made")
    print("Generating yellow thread path with" , lengths[2]*ThreadLen , "meters of thread")
    globals()['Ypath'] = generateThreadPath('Y', lengths[2]*ThreadLen*1000, img)
    print(len(Ypath), "yellow connections made")
    print("Generating black thread path with" , lengths[3]*ThreadLen , "meters of thread")
    globals()['Kpath'] = generateThreadPath('K', lengths[3]*ThreadLen*1000, img)
    print(len(Kpath), "black connections made")

    print("Drawing output image")
    drawThreads(newsize[0])

    img.save('outputFiles/result.png')

#draw the threads in a even, but random order
def drawThreads(canvasEdge):
    s=canvasEdge//2
    img = Image.new("RGB", (canvasEdge, canvasEdge), color="white")

    while len(Cpath) > 0 or len(Mpath) > 0 or len(Ypath) > 0 or len(Kpath) > 0:
        totalThreads = len(Cpath) + len(Mpath) + len(Ypath) + len(Kpath)
        code = random.randrange(totalThreads)
        if code < len(Cpath):
            nail1 = Cpath.pop(0)
            nail2 = Cpath.pop(0)
            shape = [(nail1[0]+s, nail1[1]+s), (nail2[0]+s, nail2[1]+s)]
            img1 = ImageDraw.Draw(img)  
            img1.line(shape, fill ="cyan", width = 0)
            if len(Cpath) != 0:
                Cpath.insert(0, nail2)
        elif code < len(Cpath)+len(Mpath):
            nail1 = Mpath.pop(0)
            nail2 = Mpath.pop(0)
            shape = [(nail1[0]+s, nail1[1]+s), (nail2[0]+s, nail2[1]+s)]
            img1 = ImageDraw.Draw(img)  
            img1.line(shape, fill ="magenta", width = 0)
            if len(Mpath) != 0:
                Mpath.insert(0, nail2)
        elif code < len(Cpath)+len(Mpath)+len(Ypath):
            nail1 = Ypath.pop(0)
            nail2 = Ypath.pop(0)
            shape = [(nail1[0]+s, nail1[1]+s), (nail2[0]+s, nail2[1]+s)]
            img1 = ImageDraw.Draw(img)  
            img1.line(shape, fill ="yellow", width = 0)
            if len(Ypath) != 0:
                Ypath.insert(0, nail2)
        elif code < len(Cpath)+len(Mpath)+len(Ypath)+len(Kpath):
            nail1 = Kpath.pop(0)
            nail2 = Kpath.pop(0)
            shape = [(nail1[0]+s, nail1[1]+s), (nail2[0]+s, nail2[1]+s)]
            img1 = ImageDraw.Draw(img)  
            img1.line(shape, fill ="black", width = 0)
            if len(Kpath) != 0:
                Kpath.insert(0, nail2)

        for nail in fieldNails:
            draw = ImageDraw.Draw(img)
            draw.ellipse((nail[0]+s-5, nail[1]+s-5, nail[0]+s+5, nail[1]+s+5), fill="red") #"6" determines nail-head radius on drawing
    
    img.save("outputFiles/ThreadImage-"+str(numOfEdgeNails)+"-"+str(numOfFieldNails)+"-"+str(minDistanceBetweenNails)+"-"+str(ThreadLen)+"-"+str(neighborExclusion)+"-"+str(minDistanceFNailFromEdge)+"-"+str(proportionOfThreadInk)+".png")




#Write a series of instructions to an output file, detailing the order inwhich to visit nails with one thread
def generateThreadPath(color, length, image):
    cmykImage = convertToCMYK(image)
    s = int(len(cmykImage)**0.5)

    if (color == 'C'):
        CMYKmode = 0
    elif (color == 'M'):
        CMYKmode = 1
    elif (color == 'Y'):
        CMYKmode = 2
    elif (color == 'K'):
        CMYKmode = 3

    tPath = [edgeNails[0]]
    lineScores = []
    while length > 0:
        bestNextNailYet = (0,0)
        bestNextNailScore = 0
        bestNailsLine = []

        for nail in edgeNails:
            if (math.dist(tPath[-1], nail) < neighborExclusion*minDistanceBetweenNails):
                continue
            line = list(bresenham(tPath[-1][0], tPath[-1][1], nail[0], nail[1]))
            lineScores.clear()
            for point in line: # find the average pigment removed (use median instead?)
                lineScores.append(cmykImage[xYToPixelIndex(point[0], point[1], s)][CMYKmode])
            averagePigRemoved = sum(lineScores)/len(lineScores)
            if (averagePigRemoved > bestNextNailScore):
                bestNextNailScore = averagePigRemoved
                bestNextNailYet = nail
                bestNailsLine = line

        for nail in fieldNails:
            line = list(bresenham(tPath[-1][0], tPath[-1][1], nail[0], nail[1]))
            lineScores.clear()
            for point in line:
                lineScores.append(cmykImage[xYToPixelIndex(point[0], point[1], s)][CMYKmode])
            averagePigRemoved = sum(lineScores)/len(lineScores)/1.5 #Increasing this "1.5" discourages making connections to field nails
            if (averagePigRemoved > bestNextNailScore):
                bestNextNailScore = averagePigRemoved
                bestNextNailYet = nail
                bestNailsLine = line

        length = length - (math.dist(tPath[-1], bestNextNailYet) + 0.1)
        tPath.append(bestNextNailYet)

        for point in bestNailsLine: #depigment the image
            index = xYToPixelIndex(point[0], point[1], s)
            cmykImage[index][CMYKmode] = cmykImage[index][CMYKmode]*(1.0-proportionOfThreadInk)
    
    file = open("outputFiles/"+color+"-out.txt", "w")
    for i,nail in enumerate(tPath):
        file.write(str(tPath[i]) + "-" + str(getNailNum(nail)))
    file.close()
    return tPath

#Each nail is assigned a number, given an xy coordinate pair, return the corresponding nail number
def getNailNum(xy):
    for i,nail in enumerate(edgeNails):
        if (nail == xy):
            return i

    for i,nail in enumerate(fieldNails):
        if (nail == xy):
            return i+numOfEdgeNails
    print("Uh oh")
    return -1

#converts rgba Image to cmyk np.array
def convertToCMYK(image):
    npImage=np.array(image).reshape(-1,4)
    newimage = []
    for pixel in npImage:
        if (pixel[3] == 0):
            newimage.append([0,0,0,0])
            continue

        rPrime = pixel[0]/255
        gPrime = pixel[1]/255
        bPrime = pixel[2]/255

        black = 1-(max(rPrime, gPrime, bPrime)) #complement of the brightest channel
        if (black == 1):
            newimage.append([0,0,0,255])
            continue
        cyan = (1-rPrime-black)/(1-black)
        magenta = (1-gPrime-black)/(1-black)
        yellow = (1-bPrime-black)/(1-black)
        newimage.append([cyan, magenta, yellow, black])


    return newimage

#returns 4 value list, in order CMYK of how long each thread is entitled to, based off color hues in image
def determineLengths(img):
    npImage=np.array(img).reshape(-1,4)
    lengths = [0,0,0,0] #CMYK

    for pixel in npImage:
        if (pixel[3] == 0):
            continue
        if (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0):
            continue
        rPrime = pixel[0]/255
        gPrime = pixel[1]/255
        bPrime = pixel[2]/255

        black = 1-(max(rPrime, gPrime, bPrime))
        cyan = (1-rPrime-black)/(1-black)
        magenta = (1-gPrime-black)/(1-black)
        yellow = (1-bPrime-black)/(1-black)

        lengths[0] += cyan
        lengths[1] += magenta
        lengths[2] += yellow
        lengths[3] += black

    total = sum(lengths)
    lengths[0] = lengths[0]/total
    lengths[1] = lengths[1]/total
    lengths[2] = lengths[2]/total
    lengths[3] = lengths[3]/total

    return lengths

# A nail is just a coordinate in the XY plane
# populates the global variable
def placeFieldNails(image):
    npImage=np.array(image)
    s = len(npImage)
    globals()['fieldNails'].clear()

    # Create slighty smaller alpha layer with circle, prevents field nails from being placed near border
    alpha = Image.new('L', (s,s), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([minDistanceFNailFromEdge,minDistanceFNailFromEdge,s-minDistanceFNailFromEdge,s-minDistanceFNailFromEdge],0,360,fill=255)
    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)
    # Add alpha layer to RGB
    #print(npAlpha)
    npImage=np.dstack((npImage,npAlpha)).reshape(-1,5)

    #now we make a contrast map. An array where each entry is a scalar representing how much contrast there is between it's pixel and it's pixel's neighbors
    contrastMap = []
    for pindex, pixel in enumerate(npImage):
        if (pixel[4] == 0):
            contrastMap.append(0.0)
        else:
            contrastMap.append(determineContrast(pindex, npImage, s))

    for nail in range(numOfFieldNails):
        bestPixelLocation = 0
        bestPixelScore = 0.0
        for pIndex, pScore in enumerate(contrastMap):
            if (pScore > bestPixelScore and not withinProximity(pixelIndexToXY(pIndex, s))): #if a pixel is away from other field nails and has the highest contrast
                bestPixelScore = pScore
                bestPixelLocation = pIndex
        globals()['fieldNails'].append(pixelIndexToXY(bestPixelLocation, s))
        
#Assumes 255 alpha
#finds the difference in color of a pixel
#might be expanded to check more than just surrounding 12
def determineContrast(pIndex, image, s):
    totalDist = 0.0
    R = image[pIndex][0]
    G = image[pIndex][1]
    B = image[pIndex][2]
    #adjacent pixels
    totalDist =+ math.dist((R,G,B), (image[pIndex-1][0],image[pIndex-1][1],image[pIndex-1][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+1][0],image[pIndex+1][1],image[pIndex+1][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex-s][0],image[pIndex-s][1],image[pIndex-s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+s][0],image[pIndex+s][1],image[pIndex+s][2]))

    #diagonal pixels
    totalDist =+ math.dist((R,G,B), (image[pIndex-1-s][0],image[pIndex-1-s][1],image[pIndex-1-s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+1-s][0],image[pIndex+1-s][1],image[pIndex+1-s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex-1+s][0],image[pIndex-1+s][1],image[pIndex-1+s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+1+s][0],image[pIndex+1+s][1],image[pIndex+1+s][2]))

    #adjacent+1 pixels
    totalDist =+ math.dist((R,G,B), (image[pIndex-2][0],image[pIndex-2][1],image[pIndex-2][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+2][0],image[pIndex+2][1],image[pIndex+2][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex-2*s][0],image[pIndex-2*s][1],image[pIndex-2*s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+2*s][0],image[pIndex+2*s][1],image[pIndex+2*s][2]))

    #adjacent+2 pixels
    totalDist =+ math.dist((R,G,B), (image[pIndex-3][0],image[pIndex-3][1],image[pIndex-3][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+3][0],image[pIndex+3][1],image[pIndex+3][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex-3*s][0],image[pIndex-3*s][1],image[pIndex-3*s][2]))
    totalDist =+ math.dist((R,G,B), (image[pIndex+3*s][0],image[pIndex+3*s][1],image[pIndex+3*s][2]))
    return totalDist

#returns whether or not a given coordinate is within the minimum nail distance of any field nail
def withinProximity(point):
    for nail in fieldNails:
        if (math.dist(point, nail) < minDistanceBetweenNails):
            return True
    
    return False

# A nail is just a coordinate in the XY plane
# populates the global variable
def placeEdgeNails(radius):
    degPerNail = 360/numOfEdgeNails
    globals()['edgeNails'].clear()

    for i in range(numOfEdgeNails):
        theta = math.radians(i*degPerNail)
        globals()['edgeNails'].append((int(radius*math.cos(theta)), int(radius*math.sin(theta)))) #intentionally rounds down to keep nails on colored pixels

# returns a cropped version the given PIL.Image so it is circular, resizes rectangular images into squares
def cropToCircle(image):
    h,w=image.size

    # Open the input image as numpy array, convert to RGB
    npImage=np.array(image)

    # Create same size alpha layer with circle
    alpha = Image.new('L', (h,w), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0,0,h,w],0,360,fill=255)

    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)

    # Add alpha layer to RGB
    npImage=np.dstack((npImage,npAlpha))

    return Image.fromarray(npImage)

#converts (x,y) tuple to an index in the parent image. s == size length of image
def xYToPixelIndex(x, y, s):
    return (x+s//2)+s*(y+s//2)

#converts an index in the parent image to a (x,y) coordinate. s == size length of image
def pixelIndexToXY(index, s):
    return (index%s-s//2, index//s-s//2)

if __name__ == "__main__":
    main()