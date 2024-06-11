
#  main.py

# Import the required libraries

from tkinter import Tk, Label, Button, Radiobutton, StringVar, Frame, Grid, OptionMenu
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import shutil
import os

import cv2
from PIL import Image, ImageTk, ImageDraw
from image_functions import *# Print the h, s, and v values for each pixel

from file_functions_2 import *
from random_forest import *
from segment_class import *
from dominant import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tkinter as tk
#from image_processor import ImageProcessor
from data_handler import *
from image_processing_app import ImageProcessingApp
import matplotlib.colors as mcolors
 

def create_gradient(width, height, segments):
    gradient_colors = generate_color_colors((0,0,0),(255,255,255),3)
    print(gradient_colors)
    color_width = int(width / segments)
    
    gradient_image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(gradient_image)
    
    for i, color in enumerate(gradient_colors):
        x0 = int(i * color_width)
        x1 = int((i + 1) * color_width)
        print(f'{x0},0, {x1}, {height}')
        draw.rectangle([x0, 0, x1, height], fill=color)
    
    return gradient_image

# Create the main window

window = tk.Tk()
app = ImageProcessingApp(window)
window.mainloop()

