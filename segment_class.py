

from scipy import ndimage
import time
import wx
from PIL import Image, ImageStat
import cv2 as cv2
import numpy as np
import math
import sys
import os
import matplotlib.pyplot as plt
from skimage.morphology import rectangle
import skimage.filters as filters
from skimage import img_as_ubyte
from image_functions import *
from skimage import io,img_as_float,transform
from skimage.restoration import denoise_nl_means,estimate_sigma
from skimage.util import img_as_ubyte
#from zone_class import *
import shutil
from scipy.ndimage import convolve
#import imquality.brisque as brisque
import sqlite3
from timeit import default_timer as timer
import exifread




def dehaze(img,img_gs):

    total = 0
    mean =np.average(img_gs)
    Channels = cv2.split(img)

    # Estimate Airlight
    windowSze = 3
    AirlightMethod = 'fast'
    A = Airlight(img, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 45
    C0 = 25       # Default value = 20 (as recommended in the paper)
    C1 = 260        # Default value = 300 (as recommended in the paper)
    Transmission = BoundCon(img, A, C0, C1, windowSze)                  #   Computing the Transmission using equation (7) in the paper

    # Refine estimate of transmission
    regularize_lambda = 1     # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value,
    # the closer to the original patch wise transmission
    sigma = 0.5
    Transmission = CalTransmission(img, Transmission, regularize_lambda, sigma)     # Using contextual information

    # Perform DeHazing
    return removeHaze(img, Transmission, A, 0.85)

def Airlight(HazeImg, AirlightMethod, windowSize):
    if(AirlightMethod.lower() == 'fast'):
        A = []
        if(len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                kernel = np.ones((windowSize, windowSize), np.uint8)
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)
                A.append(int(minImg.max()))
        else:
            kernel = np.ones((windowSize, windowSize), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            A.append(int(minImg.max()))
    return(A)

def BoundCon(HazeImg, A, C0, C1, windowSze):
    if(len(HazeImg.shape) == 3):

        t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float)) / (A[0] - C0),
                         (HazeImg[:, :, 0].astype(np.float) - A[0]) / (C1 - A[0]))
        t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float)) / (A[1] - C0),
                         (HazeImg[:, :, 1].astype(np.float) - A[1]) / (C1 - A[1]))
        t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float)) / (A[2] - C0),
                         (HazeImg[:, :, 2].astype(np.float) - A[2]) / (C1 - A[2]))

        MaxVal = np.maximum(t_b, t_g, t_r)
        transmission = np.minimum(MaxVal, 1)
    else:
        transmission = np.maximum((A[0] - HazeImg.astype(np.float)) / (A[0] - C0),
                         (HazeImg.astype(np.float) - A[0]) / (C1 - A[0]))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((windowSze, windowSze), np.float)
    transmission = cv2.morphologyEx(transmission, cv2.MORPH_CLOSE, kernel=kernel)
    return(transmission)

def CalTransmission(HazeImg, Transmission, regularize_lambda, sigma):
    rows, cols = Transmission.shape

    KirschFilters = LoadFilterBank()

    # Normalize the filters
    for idx, currentFilter in enumerate(KirschFilters):
        KirschFilters[idx] = KirschFilters[idx] / np.linalg.norm(currentFilter)

    # Calculate Weighting function --> [rows, cols. numFilters] --> One Weighting function for every filter
    WFun = []
    for idx, currentFilter in enumerate(KirschFilters):
        WFun.append(CalculateWeightingFunction(HazeImg, currentFilter, sigma))

    # Precompute the constants that are later needed in the optimization step
    tF = np.fft.fft2(Transmission)
    DS = 0

    for i in range(len(KirschFilters)):
        D = psf2otf(KirschFilters[i], (rows, cols))
        DS = DS + (abs(D) ** 2)

    # Cyclic loop for refining t and u --> Section III in the paper
    beta = 1                    # Start Beta value --> selected from the paper
    beta_max = 2**8             # Selected from the paper --> Section III --> "Scene Transmission Estimation"
    beta_rate = 2*np.sqrt(2)    # Selected from the paper

    while(beta < beta_max):
        gamma = regularize_lambda / beta

        # Fixing t first and solving for u
        DU = 0
        for i in range(len(KirschFilters)):
            dt = circularConvFilt(Transmission, KirschFilters[i])
            u = np.maximum((abs(dt) - (WFun[i] / (len(KirschFilters)*beta))), 0) * np.sign(dt)
            DU = DU + np.fft.fft2(circularConvFilt(u, cv2.flip(KirschFilters[i], -1)))

        # Fixing u and solving t --> Equation 26 in the paper
        # Note: In equation 26, the Numerator is the "DU" calculated in the above part of the code
        # In the equation 26, the Denominator is the DS which was computed as a constant in the above code

        Transmission = np.abs(np.fft.ifft2((gamma * tF + DU) / (gamma + DS)))
        beta = beta * beta_rate
    return(Transmission)

def LoadFilterBank():
    KirschFilters = []
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, 5],   [-3, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [-3, 0, -3],  [5, 5, 5]]))
    KirschFilters.append(np.array([[-3, -3, -3],   [5, 0, -3],   [5, 5, -3]]))
    KirschFilters.append(np.array([[5, -3, -3],    [5, 0, -3],   [5, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, -3],     [5, 0, -3],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[5, 5, 5],      [-3, 0, -3],  [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, 5, 5],     [-3, 0, 5],   [-3, -3, -3]]))
    KirschFilters.append(np.array([[-3, -3, 5],    [-3, 0, 5],   [-3, -3, 5]]))
    KirschFilters.append(np.array([[-1, -1, -1],   [-1, 8, -1],  [-1, -1, -1]]))
    return(KirschFilters)

def CalculateWeightingFunction(HazeImg, Filter, sigma):

    # Computing the weight function... Eq (17) in the paper

    HazeImageDouble = HazeImg.astype(float) / 255.0
    if(len(HazeImg.shape) == 3):
        Red = HazeImageDouble[:, :, 2]
        d_r = circularConvFilt(Red, Filter)

        Green = HazeImageDouble[:, :, 1]
        d_g = circularConvFilt(Green, Filter)

        Blue = HazeImageDouble[:, :, 0]
        d_b = circularConvFilt(Blue, Filter)

        WFun = np.exp(-((d_r**2) + (d_g**2) + (d_b**2)) / (2 * sigma * sigma))
    else:
        d = circularConvFilt(HazeImageDouble, Filter)
        WFun = np.exp(-((d ** 2) + (d ** 2) + (d ** 2)) / (2 * sigma * sigma))
    return(WFun)

def circularConvFilt(Img, Filter):
    FilterHeight, FilterWidth = Filter.shape
    assert (FilterHeight == FilterWidth), 'Filter must be square in shape --> Height must be same as width'
    assert (FilterHeight % 2 == 1), 'Filter dimension must be a odd number.'

    filterHalsSize = int((FilterHeight - 1)/2)
    rows, cols = Img.shape
    PaddedImg = cv2.copyMakeBorder(Img, filterHalsSize, filterHalsSize, filterHalsSize, filterHalsSize, borderType=cv2.BORDER_WRAP)
    FilteredImg = cv2.filter2D(PaddedImg, -1, Filter)
    Result = FilteredImg[filterHalsSize:rows+filterHalsSize, filterHalsSize:cols+filterHalsSize]

    return(Result)

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def removeHaze(HazeImg, Transmission, A, delta):
    '''
    :param HazeImg: Hazy input image
    :param Transmission: estimated transmission
    :param A: estimated airlight
    :param delta: fineTuning parameter for dehazing --> default = 0.85
    :return: result --> Dehazed image
    '''

    # This function will implement equation(3) in the paper
    # " https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.pdf "

    epsilon = 0.00001
    Transmission = pow(np.maximum(abs(Transmission), epsilon), delta)

    HazeCorrectedImage = HazeImg
    if(len(HazeImg.shape) == 3):
        for ch in range(len(HazeImg.shape)):
            temp = ((HazeImg[:, :, ch].astype(float) - A[ch]) / Transmission) + A[ch]
            temp = np.maximum(np.minimum(temp, 255), 0)
            HazeCorrectedImage[:, :, ch] = temp
    else:
        temp = ((HazeImg.astype(float) - A[0]) / Transmission) + A[0]
        temp = np.maximum(np.minimum(temp, 255), 0)
        HazeCorrectedImage = temp
    return(HazeCorrectedImage)

def create_grids(display_width, display_height, rows, columns):
    c_y = [(x * display_height / columns) for x in range(columns)]
    c_x = [(x * display_width / rows) for x in range(rows)]
    NW = [(int(y), int(x)) for x in c_x for y in c_y]
    zone_width = int(display_width / len(c_x))
    zone_height = int(display_height / len(c_y))
    return NW,zone_width,zone_height


def nothing(h):
    pass


class Thumbnail:

    def __init__(self, hue, saturation, value, img, name,delta_s,delta_v):
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.image = img
        self.name = name
        self.change_s = delta_s
        self.change_v = delta_v
        self.bitmap = self.convert_to_bitmap()


    def convert_to_bitmap(self):
        pil = cv2.cvtColor(self.image.astype("uint8"), cv2.COLOR_BGR2RGB)
        h, w = pil.shape[:2]
        bitmap = wx.BitmapFromBuffer(w, h, pil)
        return bitmap


    def apply_hsv_filter(self, original_image, hsv_filter=None):
        # convert image to HSV
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # add/subtract saturation and value
        h, s, v = cv2.split(hsv)
        s = shift_channel(s, self.bars.sAdd)
        s = shift_channel(s, -self.bars.sSub)
        v = shift_channel(v, hsv_filter.vAdd)
        v = shift_channel(v, -hsv_filter.vSub)
        hsv = cv2.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        # lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        # upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Apply the thresholds
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv2.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

class Hsv_Filter:

    def __init__(self, hMin=None, sMin=None, vMin=None, hMax=None, sMax=None, vMax=None, 
                    sAdd=None, sSub=None, vAdd=None, vSub=None):
        self.hMin = hMin
        self.sMin = sMin
        self.vMin = vMin
        self.hMax = hMax
        self.sMax = sMax
        self.vMax = vMax
        self.sAdd = sAdd
        self.sSub = sSub
        self.vAdd = vAdd
        self.vSub = vSub






class File_Name:
    def __init__(self, f_id,full_path):

        self.f_id = f_id
        self.f_root = full_path[0]
        self.full_path = full_path
        # Using os.path.dirname to get the directory part of the path
        directory = os.path.dirname(full_path)
        # Using os.path.basename to get the base part of the path (file or last directory)
        self.base_part = os.path.basename(directory)


        head, self.file_name = os.path.split(self.full_path)
        self.extension = (os.path.splitext(self.full_path)[1])

class Record_Navigator:
    def __init__(self,t_recs):

        self.f_rec = 0
        self.c_rec = self.f_rec
        self.t_recs = t_recs
        self.prev_c_rec = None
        self.first = True
        self.last = False
        self.initialize = True


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper



class FileName:
    def __init__(self, f_id,r):

        self.f_id = f_id
        self.f_root = r[0]
        self.full_path = r
        head, self.file_name = os.path.split(r)
        self.extension = (os.path.splitext(r)[1])

class CameraSettings:
    def __init__(self, full_path):

            self.exposure = 0
            self.fstop = 0
            self.iso = 0
            self.focal_length = 0
            self.bodyserialnumber = []
            self.datetimeoriginal = []
            self.get_exif_data(full_path)
    
    def get_exif_data(self, full_path):
        im = open(full_path,'rb')
        tags = exifread.process_file(im)
        for tag in tags.keys():
            if tag in "EXIF ExposureTime":
                result = str(tags[tag])
                if '/' in result:
                    new_result, x = result.split('/')
                    self.exposure = int(new_result)/int(x)
                else:
                    self.exposure = int(result)    

            if tag in "EXIF FNumber":
                result = str(tags[tag])
                if '/' in result:
                    new_result, x = result.split('/')
                    self.fstop = int(new_result)/int(x)
                else:
                    self.fstop = int(result)    

            if tag in "EXIF ISOSpeedRatings":
                result = str(tags[tag])
                self.iso = int(result)

            if tag in 'EXIF DateTimeOriginal':
                result = str(tags[tag])
                self.datetimeoriginal = result
                #print(self.datetimeoriginal)

            if tag in "EXIF BodySerialNumber":
                result = str(tags[tag])
                self.bodyserialnumber = result
                #print(self.bodyserialnumber)

            if tag in "EXIF FocalLength":
                result = str(tags[tag])
                self.focal_length = result
                #print(self.focal_length)






class ImageStatistics:
    def __init__(self, image_sharp,image_gs):

        self.get_brightness(image_sharp)

        self.get_contrast(image_gs)

        self.haze_factor = self.brightness/self.contrast

        self.get_hough_lines(image_gs)

        # self.get_hough_circles(image_gs)    

        self.get_harris(image_gs)
        
        self.get_contours(image_gs)

        self.get_laplacian(image_gs)

        self.get_variance(image_gs)

        self.get_b_w(image_gs)

        self.get_shv(image_gs)

        self.get_faces(image_gs)

        self.get_eyes(image_gs)

        self.get_bodies(image_gs)

    def get_hough_lines(self,image_gs):

        line_length = 800
        edges = cv2.Canny(image_gs, 125, 350, apertureSize=3)
        success = False
        while not success:
            lines = cv2.HoughLines(edges, 1, np.pi / 180, line_length)
            try:
                if len(lines) > 0:
                    self.hough_info = [len(lines),line_length]
                    #print(self.hough_info)
                    success = True
            except:
                line_length = line_length -100
                if line_length < 20:
                    self.hough_info = [0,20]
                    success = True

    def get_hough_circles(self,image_gs):

    # detect circles in the image
        circles = cv2.HoughCircles(image_gs, cv2.HOUGH_GRADIENT, 1.2, 100)
        # ensure at least some circles were found

        if circles is not None:
            self.hough_circles = len(circles)
        else:
            self.hough_circles = 0    


    def get_harris(self, image_gs):

    # find Harris corners
        gray = np.float32(image_gs)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        self.harris_corners = len(corners)


    def get_contours(self, image_gs):

        edged = cv2.Canny(image_gs, 50, 150, 3)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            self.contour_info = (len(contours),0)
        except:
            self.contour_info =  (0,0)


    def get_laplacian(self, image_gs):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        self.laplacian = cv2.Laplacian(image_gs, cv2.CV_64F,5).var()

    def get_variance(self, image_gs):

        try:
            im_cropped_2 = image_gs
        except:
            im_cropped_2 = self.segment

        # compute square of image
        img_sq = cv2.multiply(im_cropped_2, 2)

        # compute local mean in 5x5 rectangular region of each image
        # note: python will give warning about slower performance when processing 16-bit images
        region = rectangle(5,5)
        mean_img = filters.rank.mean(im_cropped_2, footprint=region)
        mean_img_sq = filters.rank.mean(img_sq, footprint=region)

        # compute square of local mean of img
        sq_mean_img = cv2.multiply(mean_img, mean_img)

        # compute variance using float versions of images
        var = cv2.add(mean_img_sq.astype(np.float32), - sq_mean_img.astype(np.float32))
        width = int(var.shape[1])
        height = int(var.shape[0])
        v2 = int(np.sum(var))
        v2 = int(v2/(height*width))
        self.variance = v2


    def get_b_w(self, image_gs):
    # Count pixels based on value ranges
        black_pix = np.sum(image_gs < 20)
        mid_pix = np.sum((image_gs > 19) & (image_gs < 220))
        white_pix = np.sum(image_gs > 219)
        self.b_w = (black_pix,mid_pix,white_pix)  


    def get_shv(self, image_gs):

        width, height = image_gs.shape
        pix = image_gs

        vs = []
        for y in range(height):
            row = [pix[x,y] for x in range(width)]
            mean = sum(row)/width
            variance = sum([(x-mean)**2 for x in row])/width
        vs.append(variance)
        self.shv = sum(vs)/height

    def get_faces(self, image_gs):

    # Path to the Haar cascade XML file
        cascade_path = "C:/Program Files/Sublime Text/haarcascade_frontalface_default.xml"

        # Load the Haar cascade classifier
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
        self.faces = len(faces)   
     
    def get_eyes(self, image_gs):

        # Path to the Haar cascade XML file
        cascade_path = "C:/Program Files/Sublime Text/haarcascade_mcs_eyepair_big.xml"

        # Load the Haar cascade classifier
        eye_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Detect eyes in the image
        eyes = eye_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
        self.eyes = len(eyes)   

    def get_bodies(self, image_gs):

        # Path to the Haar cascade XML file
        cascade_path = "C:/Program Files/Sublime Text/haarcascade_fullbody.xml"

        # Load the Haar cascade classifier
        body_cascade = cv2.CascadeClassifier(cascade_path)

        # Detect bodies in the image
        bodies = body_cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=10, minSize=(50,50))
        self.bodies = len(bodies)   

    def get_brightness(self, image_sharp):

        if len(image_sharp.shape) == 3:
            hsv = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2HSV)
            # print(f'Brightness: {hsv[...,2].mean()}')
            self.brightness = hsv[...,2].mean()
            print(self.brightness)

    def get_contrast(self, image_gs):
        # print(f'Contrast: {self.image_gs.std()}')
        self.contrast = image_gs.std()


 


class ColorImage:


    def __init__(self, r_src, image_id=None, fname=None,  scale_factor = 1, original_height = 0, original_width = 0):

            # print(f'ID = {image_id}\nFile Name = {fname}')            
            # cv2.imshow("Display Image", r_src)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            self.image_id = image_id
            self.fname = FileName(image_id,fname)
            self.get_classification()
            self.image = r_src
            self.image_new = r_src
            self.original_image_height = original_height
            self.original_image_width = original_width
            if original_height > original_width:
                self.orientation = 'Portrait'
            else:
                self.orientation = 'Landscape'

            self.scale_factor = (scale_factor, 1/scale_factor)
            self.image_gs = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            self.image_dnz = cv2.fastNlMeansDenoisingColored(self.image.astype(np.uint8),None,2,2,7,21)               
            self.sharpen()
            self.RODS_image = self.image_sharp

            self.image_stats = ImageStatistics(self.image_sharp, self.image_gs)
            self.camera_settings = CameraSettings(self.fname.full_path)



            # print(f'OIH {self.original_image_height} OIW {self.original_image_width} SF {self.scale_factor}')
            #obj = Color_Image()
            # methods = [method for method in dir(Color_Image) if callable(getattr(Color_Image, method)) and not method.startswith("__")]
            #print(methods)  # ['method1', 'method2']

            # cv2.imshow("RODS Image", self.image_sharp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


    def get_classification(self):
        if os.path.basename(self.fname.full_path)[0] in ("G","B", "A", "D"):
            self.classification =  os.path.basename(self.fname)[0]
        else:
            self.classification =  "U"

    def sharpen(self):

        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])

        self.image_sharp = cv2.filter2D(src= self.image_dnz, ddepth=-1, kernel = kernel)



    def convertToBinaryData(self):

        is_success, im_buf_arr = cv2.imencode(".jpg", self.image.astype(np.uint8))
        byte_im = im_buf_arr.tobytes()

        is_success, im_buf_arr = cv2.imencode(".jpg", self.image_gs)
        byte_im_gs = im_buf_arr.tobytes()

        is_success, im_buf_arr = cv2.imencode(".jpg", self.image_dnz.astype(np.uint8))
        byte_im_dnz = im_buf_arr.tobytes()

        is_success, im_buf_arr = cv2.imencode(".jpg", self.image_dnz.astype(np.uint8))
        byte_im_dhz = im_buf_arr.tobytes()

        is_success, im_buf_arr = cv2.imencode(".jpg", self.image_sharp.astype(np.uint8))
        byte_im_sharp = im_buf_arr.tobytes()

        return [byte_im, byte_im_gs, byte_im_dnz, byte_im_dhz, byte_im_sharp]

