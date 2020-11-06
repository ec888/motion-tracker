from __future__ import division
import numpy as np
import cv2
import pickle
import PathHelper
from functools import reduce

from sys import argv

DEBUG = False
###HERE ARE THE PARAMETERS:::
####################################PARAMETERS FOR ANTSY.MP4
AVG_ANT_SIZE = 10
MIN_CONTOUR_SIZE = 10  # 1/3 of the average size
MAX_CONTOUR_SIZE = 100  # 3x of the average size

KERN_SIZE = 8
PATH_SEARCH_RADIUS = 20
THRESHOLD_AT = 127
MAX_ANTS_AT_A_TIME = 75
MINIMUM_PATH_SIZE = 25
MINIMUM_PATH_STD = 3 * PATH_SEARCH_RADIUS
WRITE_TO_FILE = True
###END PARAMETERS

video_outputfile = "default_output.avi"

def avgit(x):
    return x.sum(axis=0)/np.shape(x)[0]
def plotp(p,mat,color=0):
    mat[p[0,1],p[0,0]] = color

if len(argv) != 3:
    cap = cv2.VideoCapture('evans1.mp4')
else:
    cap = cv2.VideoCapture(argv[1])
    video_outputfile = argv[2]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
ret, frame = cap.read()
height, width, layers = frame.shape
video_out = cv2.VideoWriter(video_outputfile, fourcc, 30, (width, height), True)  # FIXED: was color = True while printing gray image, which failed. Now we changed to outputing the original color image and it works.

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=90, detectShadows=False) # FIXED: threadhold default 16-> 90, to get rid of random noise at frame 1280+

kern = np.ones((KERN_SIZE,KERN_SIZE))/(KERN_SIZE**2)
ddepth = -1
def blur(image):
    return cv2.filter2D(image,ddepth,kern)
# def blr_thr(image, val=127):
#     return cv2.threshold(blur(image),val,255,cv2.THRESH_BINARY)[1]
def normalize(image):
    s = np.sum(image)
    if s == 0:
       return image
    return height*width* image / s

paths = []
archive = []

r = PATH_SEARCH_RADIUS

ind=-1

def drawPoligon(img, path, color):
    points = np.int32([reduce(lambda x, y: np.append(x, y, axis=0), path)])
    points = points.reshape((-1, 1, 2))  # FIXED: without reshape, cv2.polylines() failed. (assertion failure for npoints > 0)
    if DEBUG:
        print("poligon path, length:", len(points), "head: ", points[0], "end: ", points[len(points)-1])
    cv2.polylines(img, [points], False, color, thickness=2)  #FIXED: points -> [points]. Otherwise the drawing did not show, with no errors

def drawPoint(img, point, color):
    radius = 4
    LINE_TYPE_FILLED = 0
    if DEBUG:
        print("center point", point)
    t = tuple(point.astype(int))
    cv2.circle(img, t, radius, color, thickness=2)  # FIXED: Tuple instead of list is required

while(cap.isOpened()):
    ret, frame = cap.read()                         # 1. read image

    ind+=1
    if ind%2!=0: continue # skip odd frames. When frame frequency is too high, some objects might not appear to move.
    if DEBUG:
        # skip to the frames that we want to debug
        if ind<1220: continue
        if ind>1220+200: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 2. change to grayscale
    fgmask = fgbg.apply(frame)                      # 3. detect motion mask (foreground)
    mask = blur(fgmask)                             # 4. blur and threshold (remove noise?)
    ret2, mask = cv2.threshold(mask, THRESHOLD_AT, 255, cv2.THRESH_BINARY)

    #5. find contour and then center of contour
    contours, hierarchy_unused = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours = list(filter(lambda c: len(c)<MAX_CONTOUR_SIZE and len(c)>MIN_CONTOUR_SIZE, contours))
    contours = [c for c in contours if len(c)<MAX_CONTOUR_SIZE and len(c)>MIN_CONTOUR_SIZE]
    if contours is None or len(contours) == 0:
        print("Warning: no movement detected")
    if len(contours)>2*MAX_ANTS_AT_A_TIME:
        print("Warning: many objects detected at frame: ", ind)

    # 5.2 center points of the contours
    # FIXED: originally, the code was "scatter = map(avgit, contours)". But later as scatter is iterated twice
    # below, the second time it was an empty list. This may work in python 2 but not python 3, because map and filter
    # functions return iterator in python3.
    # The fix is to use list comprehension, according to the Python porting guide: https://portingguide.readthedocs.io/en/latest/iterators.html
    # An alternative is to use: "scatter = list(map(avgit, contours))"
    scatter = [avgit(c) for c in contours]

    # 6. calc paths
    filterWith = lambda x: len(x) > MINIMUM_PATH_SIZE and np.std(x) > MINIMUM_PATH_STD
    noisy = len(scatter) > MAX_ANTS_AT_A_TIME
    (toArchive, paths) = TrackPaths.extendPaths(r, paths, scatter, filterWith, noisy=(noisy), discard=False)
    paths = list(paths)
    archive += toArchive
    # img = (1 - mask)*gray
    img = frame  # FIXED: output original(color) image instead of gray image

    # 7. draw the path on img
    # 7.1 draw completed paths in magenta
    num_archives, num_paths = 0,0
    for path in archive:
        #color = 255
        drawPoligon(img, path, (255, 0, 255))
        num_archives+=1
    # 7.2 draw in-progress paths in blue
    for path in paths:
        #color = 0
        drawPoligon(img, path, (0, 127, 127))
        num_paths+=1

    if DEBUG:
        print(ind, ": completed path archives", num_archives, "active paths", num_paths)
        cv2.imshow('mask', mask)

    for c in scatter:
        drawPoint(img, c[0], (255, 0, 0))
    cv2.imshow('frame', img)
    video_out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_out.release()
cv2.destroyAllWindows()

if WRITE_TO_FILE:
    pickle.dump(archive, open('mypatharchive' + str(np.floor(1000*np.random.rand())) + '.pickle', 'wb'))