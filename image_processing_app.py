# image_processing_app

from tkinter import Tk, Label, Button, Radiobutton, StringVar, Frame, Grid, OptionMenu
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import shutil
import os
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
from image_functions import *
from file_functions_2 import *
from random_forest import *
from segment_class import *
from dominant import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
import tkinter as tk
from data_handler import *
from image_processor import ImageProcessor
from random_forest import *
from rf_class import *




def display_masks(b_result, s_result, h_result, s_v_result, inverse_s_v_result, counter):
    b_rgb_image = cv2.cvtColor(b_result, cv2.COLOR_BGR2RGB)
    s_rgb_image = cv2.cvtColor(s_result, cv2.COLOR_BGR2RGB)
    h_rgb_image = cv2.cvtColor(h_result, cv2.COLOR_BGR2RGB)
    s_v_rgb_image = cv2.cvtColor(s_v_result, cv2.COLOR_BGR2RGB)
    inverse_s_v_rgb_image = cv2.cvtColor(inverse_s_v_result, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(3, 2, figsize=(10, 12))  # Changed to 3 rows

    axs[0, 0].imshow(b_rgb_image, cmap='gray')
    axs[0, 0].set_title(f'Brightness Mask {counter}')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(s_rgb_image, cmap='gray')
    axs[0, 1].set_title(f'Saturation Mask {counter}')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(h_rgb_image, cmap='hsv')
    axs[1, 0].set_title(f'Hue Mask {counter}')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(s_v_rgb_image, cmap='gray')
    axs[1, 1].set_title(f'S_V Mask {counter}')
    axs[1, 1].axis('off')

    axs[2, 0].imshow(inverse_s_v_rgb_image, cmap='gray')  # Adjusted the indexing here
    axs[2, 0].set_title(f'Inverse S_V Mask {counter}')  # Adjusted the indexing here
    axs[2, 0].axis('off')  # Adjusted the indexing here


    plt.tight_layout()
    plt.show()


# def display_masks(b_result, s_result, h_result, s_v_result, inverse_s_v_result, counter):


    # b_rgb_image = cv2.cvtColor(b_result, cv2.COLOR_BGR2RGB)
    # s_rgb_image = cv2.cvtColor(s_result, cv2.COLOR_BGR2RGB)
    # h_rgb_image = cv2.cvtColor(h_result, cv2.COLOR_BGR2RGB)
    # s_v_rgb_image = cv2.cvtColor(s_v_result, cv2.COLOR_BGR2RGB)
    # inverse_s_v_rgb_image = cv2.cvtColor(inverse_s_v_result, cv2.COLOR_BGR2RGB)


    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # axs[0, 0].imshow(b_rgb_image, cmap='gray')
    # axs[0, 0].set_title(f'Brightness Mask {counter}')
    # axs[0, 0].axis('off')

    # axs[0, 1].imshow(s_rgb_image, cmap='gray')
    # axs[0, 1].set_title(f'Saturation Mask {counter}')
    # axs[0, 1].axis('off')

    # axs[1, 0].imshow(h_rgb_image, cmap='hsv')
    # axs[1, 0].set_title(f'Hue Mask {counter}')
    # axs[1, 0].axis('off')

    # axs[1, 1].imshow(s_v_rgb_image, cmap='gray')
    # axs[1, 1].set_title(f'S_V Mask {counter}')
    # axs[1, 1].axis('off')

    # axs[2, 1].imshow(inverse_s_v_rgb_image, cmap='gray')
    # axs[2, 1].set_title(f'Inverse S_V Mask {counter}')
    # axs[2, 1].axis('off')

    # plt.tight_layout()
    # plt.show()


# What exactly am I returning here? Given num_levels = 15

def create_brightness_ranges(num_levels):

    # Calculate the range size for each level
    range_size = 256 // num_levels

    # Create the dictionary of all (15 in this case) brightness ranges
    brightness_ranges = {}
    for i in range(num_levels):
        lower_bound = i * range_size
        upper_bound = (i + 1) * range_size - 1
        range_name = f"brightness_{i + 1}"
        brightness_ranges[range_name] = {"lower": lower_bound, "upper": upper_bound}

    return brightness_ranges


def create_hue_ranges(num_levels):

    # Calculate the range size for each level
    range_size = 180 // num_levels

    # Create the dictionary of all (15 in this case) hue ranges
    hue_ranges = {}
    for i in range(num_levels):
        lower_bound = i * range_size
        upper_bound = (i + 1) * range_size - 1
        range_name = f"hue_{i + 1}"
        hue_ranges[range_name] = {"lower": lower_bound, "upper": upper_bound}

    return hue_ranges

def create_saturation_ranges(num_levels):

    # Calculate the range size for each level
    range_size = 256 // num_levels

    # Create the dictionary of all (15 in this case) brightness ranges
    saturation_ranges = {}
    for i in range(num_levels):
        lower_bound = i * range_size
        upper_bound = (i + 1) * range_size - 1
        range_name = f"saturation_{i + 1}"
        saturation_ranges[range_name] = {"lower": lower_bound, "upper": upper_bound}

    return saturation_ranges

def create_s_v_ranges(num_levels):

    # Calculate the range size for each level
    range_size = 256 // num_levels

    # Create the dictionary of all (15 in this case) brightness ranges
    s_v_ranges = {}
    for i in range(num_levels):
        lower_bound = i * range_size
        upper_bound = (i + 1) * range_size - 1
        range_name = f"s_v_{i + 1}"
        s_v_ranges[range_name] = {"lower": lower_bound, "upper": upper_bound}

    return s_v_ranges


def create_brightness_masks(brightness_ranges, v_channel):

    # Create masks for each brightness range
    brightness_masks = {}
    for key, value in brightness_ranges.items():
        mask_name = key
        lower_bound = value["lower"]
        upper_bound = value["upper"]
        brightness_masks[mask_name] = cv2.inRange(v_channel, lower_bound, upper_bound)

    return brightness_masks

    
def create_hue_masks(hue_ranges, h_channel):

    # Create masks for each hue range
    hue_masks = {}
    for key, value in hue_ranges.items():
        mask_name = key
        lower_bound = value["lower"]
        upper_bound = value["upper"]
        hue_masks[mask_name] = cv2.inRange(h_channel, lower_bound, upper_bound)

    return hue_masks

def create_saturation_masks(saturation_ranges, s_channel):

    # Create masks for each hue range
    saturation_masks = {}
    for key, value in saturation_ranges.items():
        mask_name = key
        lower_bound = value["lower"]
        upper_bound = value["upper"]
        saturation_masks[mask_name] = cv2.inRange(s_channel, lower_bound, upper_bound)

    return saturation_masks

def create_s_v_masks(s_masks, v_masks, num_levels):
    s_v_masks = {}
    # num_levels = len(s_masks)  # Assuming the same number of levels for S and V masks
    
    for i in range(num_levels):
        s_v_masks[f"s_v_{str(i+1)}"] = cv2.bitwise_and(v_masks[f"brightness_{str(i+1)}"], s_masks[f"saturation_{str(i+1)}"])

    return s_v_masks

def create_inverse_s_v_masks(s_masks, v_masks, num_levels):
    inverse_s_v_masks = {}
    # num_levels = len(s_masks)  # Assuming the same number of levels for S and V masks
    
    for i in range(num_levels):
        inverse_s_v_masks[f"inverse_s_v_{str(i+1)}"] = cv2.bitwise_and(s_masks[f"saturation_{str(i+1)}"], v_masks[f"brightness_{str(num_levels-i)}"])

    return inverse_s_v_masks





def generate_colors(start_color, end_color, num_steps):
    r_step = (end_color[0] - start_color[0]) / (num_steps )
    g_step = (end_color[1] - start_color[1]) / (num_steps)
    b_step = (end_color[2] - start_color[2]) / (num_steps)

    # print(f'r_step = {end_color[0]} - {start_color[0]} / ({num_steps})')
    
    # print(f'r-step {r_step} g-step {g_step} b-step {b_step} ')
    gradient_colors = [(int(start_color[0] + r_step * i),
                        int(start_color[1] + g_step * i),
                        int(start_color[2] + b_step * i)) for i in range(num_steps)]
    # print(gradient_colors)
    return gradient_colors



def check_xlsx_file(full_excel_file):
    return os.path.exists(full_excel_file)


def open_file_dialog():
    # Ask the user to select a directory or individual files
    directory = filedialog.askdirectory()
    fnames = []
    if directory:
        parent_directory, data_file_name = os.path.split(directory)
        excel_file_name = "d:/Image Data Files/" + data_file_name + '.xlsx'
        sql_file_name = "d:/Image Data Files sql/" + data_file_name + '.db'
        # Check whether xlsx exists
        xlsx_exists = check_xlsx_file(excel_file_name)
        sql_exists = check_xlsx_file(sql_file_name)
        # print(f' {directory} {excel_file_name} {sql_file_name}')
        # print(f' {xlsx_exists} {sql_exists}')
        if xlsx_exists and sql_exists:
            return directory, excel_file_name, sql_file_name
        elif xlsx_exists:
             return directory, excel_file_name, None
        elif sql_exists:
             return directory, None, sql_file_name
        else:
            return directory, None, None
    

    else:
        return None, None, None
    


    def initialize(cls, dir_path):
        
        #print(f'dir_path {dir_path}\n\n\n\n')
        parent_directory, data_file_name = os.path.split(dir_path)
        cls.parent_dir = parent_directory
        
        cls.excel_path = "d:/Image Data Files/" + data_file_name + '.xlsx'
        cls.sql_path = "d:/Image Data Files sql/" + data_file_name + '.db'
        cls.table_sheet = "Sheet_1"
        # cls.parent_dir = parent_directory
        cls.accepted_dir = os.path.join(parent_directory, 'Accept')
        cls.rejected_dir = os.path.join(parent_directory, 'Reject')
        cls.aesthetics_dir = os.path.join(parent_directory, 'Aesthetics')
        cls.duplicate_dir = os.path.join(parent_directory, 'Duplicate')
        
        os.makedirs(cls.accepted_dir, exist_ok=True)
        os.makedirs(cls.rejected_dir, exist_ok=True)
        os.makedirs(cls.aesthetics_dir, exist_ok=True)
        os.makedirs(cls.duplicate_dir, exist_ok=True)

        # Check whether xlsx and sql db exists
        cls.excel_exists = os.path.exists(cls.excel_path)
        cls.sql_exists = os.path.exists(cls.sql_path)
        cls.file_stats["Valid Files"] = len(cls.get_valid_original_files())
        print("EFGTRGRKGRKGRGRG")
        cls.get_file_stats()







class ImageProcessingApp:
    key_labels = {
        # 27: 'Escape',
        # 2293760: 'End',
        # 2359296: 'Home',
        # 2490368: 'Up Arrow',
        # 2424832: 'Left Arrow', # Left
        # 2621440: 'Down Arrow',
        # 2555904: 'Right Arrow', # Right
        'a': 'Aesthetics',
        'A': 'Aesthetics',
        'b': 'Bad',
        'B': 'Bad',
        'd': 'Duplicate',
        'D': 'Duplicate',
        'g': 'Good',
        'G': 'Good',
        'u': 'Unclassified',
        'U': 'Unclassified',
        # 'v': 'Alt-v'

    }
    


    file_stats = {
        "Valid Files": 0,
        "Images": 0,
        "Good Directory Files Count": 0,
        "Bad Directory Files Count": 0,
        "Aesthetics Directory Files Count": 0,
        "Duplicate Directory Files Count": 0,
        
        "Unclassified Dataframe Images Count": 0,
        "Good Dataframe Images": 0,
        "Bad Dataframe Images": 0,
        "Aesthetics Dataframe Images Count": 0,
        "Duplicate Dataframe Images Count": 0,
        
        "Missing Good Directory": 0,
        "Invalid Good Directory":0,
        "Missing Bad Directory": 0,
        "Invalid Bad Directory":0,
        "Missing Aesthetics Directory": 0,
        "Invalid Aesthetics Directory":0,
        "Missing Duplicate Directory": 0,
        "Invalid Duplicate Directory":0

    }



    TOLERANCE = 4


    @classmethod
    def reset_classified_files(cls,filing_df):

        # Clear the contents of classified directories
        for directory in [cls.accepted_dir, cls.rejected_dir, cls.aesthetics_dir, cls.duplicate_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
            # List comprehension to create a list of file paths in the specified directory
            #file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error: {e}")
 

        # Repopulate classified directories based on images_pd DataFrame
        
        #Fill empty classification directories

        print("Fill empty classification directories")
        for index, row in filing_df.iterrows():
            classification = row['Classification']
            file_name = os.path.basename(row['File_Name'])
            destination_dir = None

            if classification == 'G':
                destination_dir = cls.accepted_dir
            elif classification == 'B':
                destination_dir = cls.rejected_dir
            elif classification == 'A':
                destination_dir = cls.aesthetics_dir
            elif classification == 'D':
                destination_dir = cls.duplicate_dir
            else:
                #print(f"Invalid classification for file: {file_name}")
                continue

            source_file_path = os.path.join(cls.parent_dir, file_name)
            destination_file_path = os.path.join(destination_dir, f"{classification}{file_name}")

            try:
                shutil.copy2(source_file_path, destination_file_path)
            except Exception as e:
                print(f"Error copying file {file_name}: {e}")

        print(f"Classified files have been reset.")


    @classmethod
    def recalculate_file_stats(cls, images_pd):
        
        # print(f'Here***************************************************************************')
        # Files in subdirectories
        good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = cls.get_subdir_files()       
        print(good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files)

        # Rows by classification in dataframe
        good_df_rows, bad_df_rows, aesthetics_df_rows, duplicate_df_rows, unclassified_df_rows = cls.get_rows_by_classification(images_pd)        

        directories = {
            "Good Directory": (good_df_rows, good_directory_files),
            "Bad Directory": (bad_df_rows, bad_directory_files),
            "Aesthetics Directory": (aesthetics_df_rows, aesthetics_directory_files),
            "Duplicate Directory": (duplicate_df_rows, duplicate_directory_files)
        }

        for directory, (df_rows, dir_files) in directories.items():
            print(directory,(len(df_rows),len(dir_files)))
            missing_files = len(df_rows) - len(dir_files)
            invalid_files = len(dir_files) - len(df_rows)
            cls.file_stats[f"Missing from {directory}"] = missing_files
            cls.file_stats[f"Invalid in {directory}"] = invalid_files
            if missing_files > 0 or invalid_files > 0:
                filing_df = images_pd[['File_Name', 'Classification']]
                return filing_df

            filing_df = pd.DataFrame()
            return filing_df


    @classmethod
    def get_rows_by_classification(cls, images_pd):
        class_counts = images_pd['Classification'].value_counts()
        
        cls.file_stats["Unclassified Dataframe Images Count"] = class_counts.get('U', 0)
        cls.file_stats["Good Dataframe Images Count"] = class_counts.get('G', 0)
        cls.file_stats["Bad Dataframe Images Count"] = class_counts.get('B', 0)
        cls.file_stats["Aesthetics Dataframe Images Count"] = class_counts.get('A', 0)
        cls.file_stats["Duplicate Dataframe Images Count"] = class_counts.get('D', 0)

        # Generate lists of file names for each row of the dataframe corrected for classification 
        good_df_rows = set(images_pd.loc[images_pd['Classification'] == 'G', 'File_Name'].apply(lambda x: 'G' + os.path.basename(x)))
        bad_df_rows = set(images_pd.loc[images_pd['Classification'] == 'B', 'File_Name'].apply(lambda x: 'B' + os.path.basename(x)))
        aesthetics_df_rows = set(images_pd.loc[images_pd['Classification'] == 'A', 'File_Name'].apply(lambda x: 'A' + os.path.basename(x)))
        duplicate_df_rows = set(images_pd.loc[images_pd['Classification'] == 'D', 'File_Name'].apply(lambda x: 'D' + os.path.basename(x)))
        unclassified_df_rows = set(images_pd.loc[images_pd['Classification'] == 'U', 'File_Name'].apply(lambda x: '' + os.path.basename(x)))

        return good_df_rows, bad_df_rows, aesthetics_df_rows,duplicate_df_rows, unclassified_df_rows




    @classmethod
    def get_valid_original_files(cls):

        valid_files = []
        # Step 1: List all files in the directory
        parent_directory_files = set(os.listdir(cls.dir_path))
        print(f'Parent Directory File from function: {parent_directory_files}')
        for file in parent_directory_files:
            if file.endswith('.JPG') and '$' not in file and 'r_' not in file:
                valid_files.append(file)
        cls.file_stats["Valid Files"] = len(valid_files)
        print(f'Number of Valid Files: {len(valid_files)}')        
        # verify_dataframe_images_with_originals(valid_files)        

        return valid_files





    # @classmethod
    # def reset_classified_files(cls,filing_df):

    #     # Clear the contents of classified directories
    #     for directory in [cls.accepted_dir, cls.rejected_dir, cls.aesthetics_dir, cls.duplicate_dir]:
    #         for filename in os.listdir(directory):
    #             file_path = os.path.join(directory, filename)
    #         # List comprehension to create a list of file paths in the specified directory
    #         #file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]

    #             try:
    #                 if os.path.isfile(file_path):
    #                     os.remove(file_path)
    #                 elif os.path.isdir(file_path):
    #                     shutil.rmtree(file_path)
    #             except Exception as e:
    #                 print(f"Error: {e}")
 

        # Repopulate classified directories based on images_pd DataFrame
        for index, row in filing_df.iterrows():
            classification = row['Classification']
            file_name = os.path.basename(row['File_Name'])
            destination_dir = None

            if classification == 'G':
                destination_dir = cls.accepted_dir
            elif classification == 'B':
                destination_dir = cls.rejected_dir
            elif classification == 'A':
                destination_dir = cls.aesthetics_dir
            elif classification == 'D':
                destination_dir = cls.duplicate_dir
            else:
                print(f"Invalid classification for file: {file_name}")
                continue

            source_file_path = os.path.join(cls.parent_dir, file_name)
            destination_file_path = os.path.join(destination_dir, f"{classification}{file_name}")

            try:
                shutil.copy2(source_file_path, destination_file_path)
            except Exception as e:
                print(f"Error copying file {file_name}: {e}")

        print(f"Classified files have been reset.")











    @classmethod
    def get_file_stats(cls):
            
        good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = cls.get_subdir_files()


    @classmethod
    def get_subdir_files(cls):
    
        all_items = os.listdir(cls.accepted_dir)
        # print([item for item in all_items if os.path.isfile(os.path.join(cls.accepted_dir, item))])
        print(f'Accepted Directory {cls.accepted_dir}')
        
        # Filter out directories and their files

        good_directory_files = [item for item in all_items if os.path.isfile(os.path.join(cls.accepted_dir, item))]
        print(f'def get_subdir_files ()')
        print(f' Good Files Count: {len(good_directory_files)}')
        print(f' Good Files: {good_directory_files}')

        
        bad_directory_files = set(os.listdir(cls.rejected_dir))
        print(f' Bad Files Count: {len(bad_directory_files)} {bad_directory_files}')
        print(f' Bad Files: {bad_directory_files}')
        

        aesthetics_directory_files = set(os.listdir(cls.aesthetics_dir))
        print(f' Aesthetics_Files Count: {len(aesthetics_directory_files)}')
        print(f' Aesthetics_Files: {aesthetics_directory_files}')

        duplicate_directory_files = set(os.listdir(cls.duplicate_dir))
        print(f' Duplicate Files Count: {len(duplicate_directory_files)}')
        print(f' Duplicate Files: {duplicate_directory_files}')
        
        cls.file_stats["Good Directory Files Count"] =  len(good_directory_files)
        cls.file_stats["Bad Directory Files Count"] = len(bad_directory_files)
        cls.file_stats["Aesthetics Directory Files Count"] = len(aesthetics_directory_files)
        cls.file_stats["Duplicate Directory Files Count"] = len(duplicate_directory_files)

        return good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files



    @classmethod
    def set_dirs(cls, dir_path):
        parent_directory, data_file_name = os.path.split(dir_path)
        print(f'Parent:{parent_directory}\nData File Name: {data_file_name}')
        cls.parent_dir = parent_directory
        cls.excel_path = "d:/Image Data Files/" + data_file_name + '.xlsx'
        cls.sql_path = "d:/Image Data Files sql/" + data_file_name + '.db'
        cls.table_sheet = "Sheet_1"
        cls.parent_dir = parent_directory
        cls.accepted_dir = os.path.join(parent_directory, 'Accept')
        cls.rejected_dir = os.path.join(parent_directory, 'Reject')
        cls.aesthetics_dir = os.path.join(parent_directory, 'Aesthetics')
        cls.duplicate_dir = os.path.join(parent_directory, 'Duplicate')
        
        os.makedirs(cls.accepted_dir, exist_ok=True)
        os.makedirs(cls.rejected_dir, exist_ok=True)
        os.makedirs(cls.aesthetics_dir, exist_ok=True)
        os.makedirs(cls.duplicate_dir, exist_ok=True)

        # Check whether xlsx and sql db exists
        cls.excel_exists = os.path.exists(cls.excel_path)
        cls.sql_exists = os.path.exists(cls.sql_path)
        cls.file_stats["Valid Files"] = len(cls.get_valid_original_files())
        cls.get_file_stats()
    '''
    
    Init.................

    '''

    def __init__(self, root):

        self.root = root
        # self.root.geometry("1000x800")
        self.root.title("BobL Images Image Processing Application")

        self.image_processor = ImageProcessor()
        self.data_handler = DataHandler()
        self.random_forest_model_instance = RandomForestModel()
        self.display_images = []
        self.file_objects = []
        # self.active_image = Color_Image()
      
        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        
        # Set the geometry of the root window to full screen
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.record_index = 0
        self.sorted_record_index = 0

        self.initial_image_flag = True
        self.num_images = 0
        


        # PIL Image
        self.tk_image = None
        
        self.image_display_type = None
        
        # ??? Image
        self.display_image = None
        

        self.landscape_grid = create_grids(1344, 2016, 5,7)
        self.portrait_grid = create_grids(1344, 896, 7,5)

        self.start_x = 0
        self.start_y = 0

        self.move_operation = False
        self.corner_click = False
        self.line_click = False
        
        self.move_start_x = 0
        self.move_start_y = 0
        self.rectangle_id = 0



        # Bind the keys to the event handler
        self.root.bind('<Right>', self.on_right_arrow_key)
        self.root.bind('<Left>', self.on_left_arrow_key)
        self.root.bind('<End>', self.on_end_key)
        self.root.bind('<Home>', self.on_home_key)
        self.root.bind('<A>', lambda event: self.on_classification_key(event, 'A'))
        self.root.bind('<a>', lambda event: self.on_classification_key(event, 'A'))
        self.root.bind('<B>', lambda event: self.on_classification_key(event, 'B'))
        self.root.bind('<b>', lambda event: self.on_classification_key(event, 'B'))
        self.root.bind('<D>', lambda event: self.on_classification_key(event, 'D'))
        self.root.bind('<d>', lambda event: self.on_classification_key(event, 'D'))
        self.root.bind('<G>', lambda event: self.on_classification_key(event, 'G'))
        self.root.bind('<g>', lambda event: self.on_classification_key(event, 'G'))






        self.initialize_gui()

    def load_image(self, file_name):
        # Load the file 
        print(file_name)
        image = cv2.imread(file_name)
        # image_height = image.shape[0]
        # image_width = image.shape[1]
        # print(f'Original Image Shape: {image_height}x{image_width}')
        return image

    def resize_image_for_display(self, O_image):
        scale_factor  = self. get_scale(O_image)
        
        aspect_ratio = O_image.shape[0]/O_image.shape[1]
        # print(f'Aspect Ratio: {aspect_ratio}')

        RO_image_height = int(O_image.shape[0] * scale_factor)
        RO_image_width = int(O_image.shape[1] * scale_factor)          

        RO_image_dim = (RO_image_width, RO_image_height)
        RO_image= cv2.resize(O_image,RO_image_dim, interpolation = cv2.INTER_AREA).astype(np.uint8)

        return (RO_image,scale_factor)

    def get_scale(self, O_image):
        
        if O_image.shape[1] < O_image.shape[0]:
            scale_factor = 896/ O_image.shape[1]
        elif O_image.shape[1] > O_image.shape[0]:
            scale_factor = 2016/ O_image.shape[1]
        else:
            scale_factor = 1

        return scale_factor


    def load_images(self):


        images_for_loading = []
        counter = 0
        enhanced_actions = False
        batch_size = 25
        num_batches = len(self.file_objects) // batch_size + 1
        total_rec_ctr = 0

        print(f'Batches: {num_batches}')


        for batch_num in range(num_batches):
            batch_files = self.file_objects[batch_num * batch_size: (batch_num + 1) * batch_size]

            for counter, file_object in enumerate(batch_files):
                fname = file_object.full_path  # Access the full path if 'file_object' has this attribute
            # for counter, fname in enumerate(batch_files):
            #     print(f'Batch Files: {batch_files}')
                O_image = self.load_image(fname)
                RO_image, scale_factor = self.resize_image_for_display(O_image)
                ctr = counter + batch_num * batch_size
                self.display_images.append(ColorImage(RO_image, ctr +1 , self.file_objects[ctr].full_path, scale_factor, O_image.shape[0], O_image.shape[1]))
                total_rec_ctr += 1
                print(f'Loaded {total_rec_ctr} of {len(self.file_objects)} Images')



    def process_images(self):
    
        # self.dir_path = self.root.menu_frame.file_frame.dir_entry['text'] 
        # self.excel_path = self.root.menu_frame.file_frame.excel_entry['text']
        # self.sql_path = self.root.menu_frame.file_frame.sql_entry['text']
            
        fnames = get_files(ImageProcessingApp.dir_path)
        
        # Create list of file objects to load
        self.file_objects = [File_Name(ctr, fname) for ctr, fname in enumerate(fnames)]
        # print(f'\n\nfile objects set\n\n')
        
        self.load_images()
        # print(f'\n\n Resized images loaded: {len(self.display_images)}\n\n')

        self.image_processor.create_dataframe(self.display_images,self.file_objects)

        # print("Image Processor process_images method", self.images_pd)


        file_types = ['excel', 'sql']
        for file_type in file_types:
            self.image_processor.save_data_from_data_handler(file_type,ImageProcessingApp.dir_path, ImageProcessingApp.excel_path,
             ImageProcessingApp.sql_path, ImageProcessingApp.table_sheet)





#     def process_image_listing(self):
    


#         # Get and read file into a list

#         # Iterate through the list

#         for each dir in dir_list

#                 parent_directory, data_file_name = os.path.split(dir)
#                 excel_file_name = "d:/Image Data Files/" + data_file_name + '.xlsx'
#                 sql_file_name = "d:/Image Data Files sql/" + data_file_name + '.db'
#                 self.dir_path = dir
#                 self.excel_path =  excel_file_name
#                 self.sql_path = sql_file_name


#         fnames = get_files(ImageProcessingApp.dir_path)
        
#         # Create list of file objects to load
#         self.file_objects = [File_Name(ctr, fname) for ctr, fname in enumerate(fnames)]
#         # print(f'\n\nfile objects set\n\n')
        
#         self.load_images()
#         # print(f'\n\n Resized images loaded: {len(self.display_images)}\n\n')

#         self.image_processor.create_dataframe(self.display_images,self.file_objects)

#         # print("Image Processor process_images method", self.images_pd)


#         file_types = ['excel', 'sql']
#         for file_type in file_types:
#             self.image_processor.save_data_from_data_handler(file_type,ImageProcessingApp.dir_path, ImageProcessingApp.excel_path,
#              ImageProcessingApp.sql_path, ImageProcessingApp.table_sheet)

# '''







    def create_file_frame(self):

        self.root.menu_frame.file_frame = Frame(self.root.menu_frame)

        label_text = ["Directory:", "Excel File:","SQL File:"]
        for counter, label in enumerate(label_text):
            tk.Label(self.root.menu_frame.file_frame, text=label).grid(row=counter, column=0, sticky="w")

        self.root.menu_frame.file_frame.dir_entry = tk.Label(self.root.menu_frame.file_frame, text="")
        self.root.menu_frame.file_frame.excel_entry = tk.Label(self.root.menu_frame.file_frame, text="")
        self.root.menu_frame.file_frame.sql_entry = tk.Label(self.root.menu_frame.file_frame, text="")

        self.root.menu_frame.file_frame.dir_entry.grid(row=0, column=1, sticky="w")
        self.root.menu_frame.file_frame.excel_entry.grid(row=1, column=1, sticky="w")
        self.root.menu_frame.file_frame.sql_entry.grid(row=2, column=1, sticky="w")

        self.root.menu_frame.file_frame.grid(row=0, column=0)

    def create_brightness_frame(self):   

        # Create a frame for the brightness adjustment widget section of the window
        self.root.menu_frame.brightness_frame = tk.Frame(self.root.menu_frame, bd=2, relief=tk.GROOVE)
        self.root.menu_frame.brightness_frame.grid(row=3, column=0, sticky="nw")  # padx=1, pady=1,

        brightness_title = Label(self.root.menu_frame.brightness_frame, text="Brightness Adjustment")
        brightness_title.grid(row=0, column=0, columnspan=2)  # Adjust the columnspan as needed

        # Create labels for the brightness levels

        # brightness_labels = ["Hue", "Contrast", "Value"]
        # #  Generate gradients for the sliders (dark to light)
        # gradient_colors = generate_colors((0,0,0),(255,255,255),3)



        '''
        Creates 15 labels        
        '''
        brightness_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11","12","13","14"]
        #  Generate gradients for the sliders (dark to light)
        gradient_colors = generate_colors((0,0,0),(255,255,255),15)
        
        # Create sliders to adjust brightness levels
        self.brightness_sliders = []
        for idx, label in enumerate(brightness_labels):
            slider_color = gradient_colors[idx]
            slider = tk.Scale(self.root.menu_frame.brightness_frame, from_=255, to=0, orient="vertical", length=200, width = 5,
                highlightthickness=0, troughcolor='#%02x%02x%02x' % slider_color)
            slider.grid(row=1, column=idx)
            tk.Label(self.root.menu_frame.brightness_frame, text=label).grid(row=2, column=idx )  #padx=1, pady=1
            self.brightness_sliders.append(slider)

        for slider in self.brightness_sliders:
            slider.config(command=self.slider_callback)

    def create_statistics_frame(self):
        # Create a frame for the statistics section
        self.root.menu_frame.statistics_frame = tk.Frame(self.root.menu_frame, bd=2, relief=tk.GROOVE)
        self.root.menu_frame.statistics_frame.grid(row=4, column=0, sticky="nw")

        # Create Treeview widget
        self.root.menu_frame.statistics_frame.stats_tv = ttk.Treeview(self.root.menu_frame.statistics_frame)
        self.root.menu_frame.statistics_frame.stats_tv["height"] = 28
        self.root.menu_frame.statistics_frame.stats_tv.pack(expand=tk.YES, fill=tk.BOTH)

        # Add columns to Treeview
        self.root.menu_frame.statistics_frame.stats_tv["columns"] = ("Value")
        self.root.menu_frame.statistics_frame.stats_tv.column("#0", width=150, minwidth=150, stretch=tk.NO)
        self.root.menu_frame.statistics_frame.stats_tv.column("Value", width=150, minwidth=100, stretch=tk.NO)
        self.root.menu_frame.statistics_frame.stats_tv.heading("#0", text="Statistic", anchor=tk.W)
        self.root.menu_frame.statistics_frame.stats_tv.heading("Value", text="Value", anchor=tk.W)

        # Get initial stats and display
        # print(f'Image Flag: {self.initial_image_flag}')
        # if self.initial_image_flag == False:
        #     self.update_stats()

    def update_stats(self):
        # Get data for the current record
        stats_pd = self.image_processor.sorted_pd.iloc[[self.sorted_record_index]]

        # Exclude certain columns
        columns_to_exclude = ['Original_Image', 'Grayscale_Image', 'Denoised_Image', 'Sharpened_Image']
        modified_stats_pd = stats_pd.drop(columns=columns_to_exclude)

        # Print the original list of columns
        # print(f"Original columns: {stats_pd.columns}")

        # Print the updated list of columns
        # print(f"Updated columns: {modified_stats_pd.columns}")

        # Clear existing items in the Treeview if there are any
        
        children = self.root.menu_frame.statistics_frame.stats_tv.get_children()
        # print(f'Children: {children}')
        if children:
            self.root.menu_frame.statistics_frame.stats_tv.delete(*children)

        # Insert data into Treeview
        for key, value in modified_stats_pd.items():
            # print(f"Inserting: {key} - {value.values[0]}")
            self.root.menu_frame.statistics_frame.stats_tv.insert("", tk.END, text=key, values=(str(value.values[0])))

        # Update the frame title with the current file name
        # file_name = modified_stats_pd["File_Name"]
        # self.root.menu_frame.statistics_frame.stats_tv.master.master.master.master.title(f"Statistics - {file_name}")

    def create_data_source_frame(self):

        # Create a frame for the data source radio buttons
        self.root.menu_frame.body_source_frame.data_source_frame = tk.Frame(self.root.menu_frame.body_source_frame, bd=2, relief=tk.GROOVE)
        self.root.menu_frame.body_source_frame.data_source_frame.grid(row=1, column=0, padx=1, pady=1, sticky="nw")

        # data_source frame contents

        # Add a label for data source
        self.root.menu_frame.body_source_frame.data_source_frame.data_source_label = tk.Label( self.root.menu_frame.body_source_frame.data_source_frame, text="Data Source:")
        self.root.menu_frame.body_source_frame.data_source_frame.data_source_label.grid(row=0, column=0, sticky="w")

        # Create radio buttons for data source
        self.source_var = tk.StringVar()
        self.root.menu_frame.body_source_frame.data_source_frame.excel_radio = Radiobutton(self.root.menu_frame.body_source_frame.data_source_frame, text="Load Excel Data",
         state= tk.DISABLED, variable=self.source_var, value="excel")
        self.root.menu_frame.body_source_frame.data_source_frame.sql_radio = Radiobutton( self.root.menu_frame.body_source_frame.data_source_frame, text="Load SQL Data",
         state= tk.DISABLED, variable=self.source_var, value="sql")
        self.root.menu_frame.body_source_frame.data_source_frame.excel_radio.grid(row=1, column=0, sticky="w")
        self.root.menu_frame.body_source_frame.data_source_frame.sql_radio.grid(row=2, column=0, sticky="w")
    
    def create_sort_order_frame(self):

        # Create a frame for the sort order radio buttons
        self.root.menu_frame.body_source_frame.sort_order_frame = Frame(self.root.menu_frame.body_source_frame, bd=2, relief=tk.GROOVE)
        self.root.menu_frame.body_source_frame.sort_order_frame.grid(row=1, column=1,  padx=2, pady=2, sticky="nw")

        # Sort Frame Contents

        # Sort options
        sort_options = ['Image_Id (Original Order)', 'Brightness','Contours','Laplacian', 'Classification (U)', 'Classification (B)','SHV',
        'd_mean_b', 'd_mean_c', 'd_mean_l', 'Orientation-P', 'Orientation-L']

        # Create a StringVar to hold the selected sorting option
        self.sort_var = tk.StringVar()
        self.sort_var.set(sort_options[0])  # Default sort option

        # Create the sorting label and dropdown menu
        self.root.menu_frame.body_source_frame.sort_order_frame.sort_label = tk.Label(self.root.menu_frame.body_source_frame.sort_order_frame, text="Sort By:")
        self.root.menu_frame.body_source_frame.sort_order_frame.sort_label.grid(row=0, column=0, sticky="w", padx=1)
        
        self.root.menu_frame.body_source_frame.sort_order_frame.sort_dropdown = tk.OptionMenu(self.root.menu_frame.body_source_frame.sort_order_frame, self.sort_var, *sort_options)
        self.root.menu_frame.body_source_frame.sort_order_frame.sort_dropdown.grid(row=0, column=1, sticky="w", padx=25, pady = 30)
        self.root.menu_frame.body_source_frame.sort_order_frame.sort_dropdown.config(width=10)  # Set the width of the dropdown menu

    def create_button_frame(self):

        # Create a frame for the button group at the bottom
        self.root.menu_frame.button_frame = tk.Frame(self.root.menu_frame)
        self.root.menu_frame.button_frame.grid(row=5, column=0, columnspan=3, pady=10,sticky="sw")


        # Button Frame contents
        # SELECT TARGET DIRECTORY

        # Create a button for selecting the directory to load
        self.root.menu_frame.button_frame.select_dir_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Select Directory",
            state=tk.NORMAL,
            command=lambda: self.on_click("A", self.root.menu_frame.body_source_frame.data_source_frame.excel_radio,
                self.root.menu_frame.body_source_frame.data_source_frame.sql_radio)
        )
        self.root.menu_frame.button_frame.select_dir_btn.grid(row=0, column=0)

        # PROCESS MULTIPLE IMAGES

        # Create a button for processing images
        self.root.menu_frame.button_frame.process_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Process Images",
            state=tk.DISABLED,
            command=lambda: self.process_images()
        )
        self.root.menu_frame.button_frame.process_btn.grid(row=0, column=1)

        # PROCESS SINGLE IMAGE

        # Create a button for processing a single image
        self.root.menu_frame.button_frame.process_single_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Process Single Image",
            state=tk.DISABLED,
            command=lambda: self.image_processor.process_images(False)
        )
        self.root.menu_frame.button_frame.process_btn.grid(row=0, column=2)
        

        # VIEW

        # Create a button for viewing images
        self.root.menu_frame.button_frame.view_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="View Images",
            state=tk.DISABLED,
            command=lambda: self.initial_display()
        )
        self.root.menu_frame.button_frame.view_btn.grid(row=0, column=3)


        # LOAD
        
        # Create a button for running the Load Data module
        self.root.menu_frame.button_frame.load_data_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Load Data",
            state=tk.DISABLED,
            command=lambda: self.image_processor.load_data_from_data_handler(ImageProcessingApp.sql_path,ImageProcessingApp.table_sheet, self.source_var.get(), self.sort_var.get())

        )
        self.root.menu_frame.button_frame.load_data_btn.grid(row=0, column=4)


        # TRAIN

        # Create a button for running the Train module
        self.root.menu_frame.button_frame.train_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Run Train",
            state=tk.NORMAL,
            command=lambda: self.rf_train()
        )
        self.root.menu_frame.button_frame.train_btn.grid(row=1, column=0)


        # TEST

        # Create a button for running the Test module
        self.root.menu_frame.button_frame.test_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Run Test",
            state=tk.DISABLED,
            command=lambda: self.rf_test()
        )
        self.root.menu_frame.button_frame.test_btn.grid(row=1, column=1)


        # DISPLAY PLOTS 1

        # Create a button for displaying plots
        self.root.menu_frame.button_frame.display_plots_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Display Plots",
            state=tk.DISABLED,
            command=lambda: self.image_processor.create_plots(['Image_ID', 'Image_ID', 'Image_ID', 'Image_ID', 'Image_ID',
                'Laplacian', 'Laplacian', 'Harris_Corners', 'Contour_Info', 'SHV'],
                ['Brightness', 'Contour_Info', 'SHV', 'Harris_Corners', 'Laplacian',
                'Brightness', 'Contour_Info','Brightness', 'Brightness', 'Brightness']
                )
            )
        self.root.menu_frame.button_frame.display_plots_btn.grid(row=2, column=0)


        # DISPLAY PLOTS 2        

        # Create a button for displaying d_mean_* plots
        self.root.menu_frame.button_frame.display_plots_2_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Display delta_d Plots",
            state=tk.NORMAL,
            command=lambda: self.image_processor.create_plots_2(['Image_ID', 'Image_ID', 'Image_ID'],
                ['d_mean_b', 'd_mean_c', 'd_mean_l']
                )
            )
        self.root.menu_frame.button_frame.display_plots_2_btn.grid(row=2, column=2)


        # DISPLAY DATA (NON FUNCTIONAL)

        # Create a button for displaying data
        self.root.menu_frame.button_frame.display_data_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Display Data",
            state=tk.NORMAL,
            command=lambda: display_data(images_pd)
        )
        self.root.menu_frame.button_frame.display_data_btn.grid(row=3, column=0)

        # CREATE FINAL IMAGES
        
        # Create a button for creating final images
        self.root.menu_frame.button_frame.create_final_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Create Final",
            state=tk.DISABLED,
            command=lambda: self.image_processor.process_and_save_accepted_images()
        )
        self.root.menu_frame.button_frame.create_final_btn.grid(row=3, column=1)

        
        # VIEW GOOD

        # Create a button to view (filter) only "good" reduced images
        self.root.menu_frame.button_frame.display_data_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="View Good Images",
            state=tk.NORMAL,
            command=lambda: self.view_good_images()
        )
        self.root.menu_frame.button_frame.display_data_btn.grid(row=2, column=1)


        # ACCEPT CROP

        # Create a button to Accept Crop
        self.root.menu_frame.button_frame.accept_crop_btn = tk.Button(
            self.root.menu_frame.button_frame,
            text="Accept Crop",
            state=tk.NORMAL,
            command=lambda: self.accept_crop()
        )
        self.root.menu_frame.button_frame.accept_crop_btn.grid(row=3, column=2)

    def initialize_gui(self):

        # Get the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set the geometry of the root window to full screen
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        # Calculate the width for menu_frame and image_frame
        menu_width = int(screen_width * 0.22)
        image_width = screen_width - menu_width

        # Create menu_frame
        self.root.menu_frame = tk.Frame(self.root, width=menu_width, height=screen_height, bg="white")
        self.root.menu_frame.grid(row=0, column=0, sticky="nsw")

        # Create image_frame
        self.root.image_frame = tk.Frame(self.root, width=image_width, height=screen_height, bg="green")
        self.root.image_frame.grid(row=0, column=1, sticky="nsew")

        # Update weight of columns to make them resizable
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)

        # Create a frame for the body section of the menu
        self.root.menu_frame.body_source_frame = tk.Frame(self.root.menu_frame, bd=2, relief=tk.GROOVE)
        self.root.menu_frame.body_source_frame.grid(row=2, column=0, padx=2, pady=2, sticky="nw")

        self.root.image_frame.canvas = tk.Canvas(self.root.image_frame, width = 2016, height = 1344) # , width=desired_width, height=desired_height
        self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        # Bind mouse events
        self.root.image_frame.canvas.bind("<ButtonPress-1>", self.on_press)
        # self.root.image_frame.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.image_frame.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.image_frame.canvas.bind("<ButtonRelease-1>", self.on_release)


        # Create File Frame for the file/directory labels
        self.create_file_frame()

        # Create Datasource Frame for the datasource radio buttons
        self.create_data_source_frame()

        # Create Sort Order Frame for the sorting listbox
        self.create_sort_order_frame()

        # Create Brightness Frame for the brightness sliders
        self.create_brightness_frame()

        # Create Statistics Frame for stat info presentation using TreeView
        self.create_statistics_frame()

        # Create Button Frame for.....buttons
        self.create_button_frame()


    def on_click(self,text,excel_radio, sql_radio):
        ImageProcessingApp.dir_path, ImageProcessingApp.excel_path, ImageProcessingApp.sql_path = open_file_dialog()
        self.root.menu_frame.file_frame.dir_entry['text'] = ImageProcessingApp.dir_path
        self.root.menu_frame.file_frame.excel_entry['text'] = ImageProcessingApp.excel_path
        self.root.menu_frame.file_frame.sql_entry['text'] = ImageProcessingApp.sql_path
        print(f'Initializing Directory {ImageProcessingApp.dir_path}')
        self.set_dirs(ImageProcessingApp.dir_path)
        
        if ImageProcessingApp.dir_path:
            print(f'Initializing Directory success')
            print(f'Process Button State: {self.root.menu_frame.button_frame.process_btn["state"]}')
            self.root.menu_frame.button_frame.process_btn['state'] = tk.NORMAL
        if ImageProcessingApp.excel_path:
            self.root.menu_frame.button_frame.load_data_btn['state'] = 'normal'
            self.root.menu_frame.button_frame.view_btn['state'] = 'normal'
            self.root.menu_frame.body_source_frame.data_source_frame.excel_radio['state']= 'normal'

        if ImageProcessingApp.sql_path:
            self.root.menu_frame.button_frame.load_data_btn['state'] = 'normal'
            self.root.menu_frame.button_frame.view_btn['state'] = 'normal'
            self.root.menu_frame.body_source_frame.data_source_frame.sql_radio['state']= 'normal'


    def slider_callback(self, *args):

        # This function will be called when any slider is moved
        # Read the values of all sliders and update the image
        brightness_adjustments = [slider.get() for slider in self.brightness_sliders]

        range_dict = create_brightness_ranges(15)

        # Create a list of tuples where each tuple contains the adjustment value and its corresponding range dictionary
        adjustments_with_ranges = [(adjustment, range_dict[f"brightness_{i+1}"]) for i, adjustment in enumerate(brightness_adjustments)]




        print(f'Brightness Values: {brightness_adjustments}')

        # print(self.__dict__)

        # cv2.imshow("ddf", self.display_image.image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()        

        # Convert self.display_image to hsv
        hsv_image =  cv2.cvtColor(self.display_image.image, cv2.COLOR_BGR2HSV)
        
        #image_adjustments = brightness_values
        self.display_image.image_new = self.modify_HSV(hsv_image, adjustments_with_ranges)

        self.cv2PIL()

        # Add the image to the Canvas
        self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        self.update_stats()
        self.initial_image_flag = False
        self.root.menu_frame.button_frame.display_data_btn.state = 'Normal'

 










 
    # def modify_HSV(self, hsv_image, image_adjustments):

    #     hue_adjust = image_adjustments[0]
    #     saturation_adjust = image_adjustments[1]
    #     value_adjust = image_adjustments[2]



    #     # Split HSV image
    #     h, s, v = cv2.split(hsv_image)

        

    #     print(f'Hue Adjust {hue_adjust} Saturation Adjust {saturation_adjust} Value Adjust {value_adjust}')

    #     # Adjust each channel
    #     h += hue_adjust
    #     s += saturation_adjust
    #     v += value_adjust

    #     # Clip the values to stay within the valid range (0 to 255)
    #     h = np.clip(h, 0, 180)
    #     s = np.clip(s, 0, 255)
    #     v = np.clip(v, 0, 255)

    #     # Merge the adjusted channels back into the HSV image
    #     modified_hsv = cv2.merge([h, s, v])

    #     print(modified_hsv)
    #     # Convert the modified HSV image back to BGR for display
    #     return cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    # '''

    def modify_HSV(self,hsv_image, adjustments_with_ranges):

        h_channel, s_channel, v_channel = cv2.split(hsv_image)

        # Step 1: Convert to a larger type for the calculation
        v_channel = v_channel.astype(np.uint32)

        # Apply modifications to v_channel here

        for adjustment, brightness_range in adjustments_with_ranges:
            # Now you can access adjustment and brightness_range individually in each iteration
            lower_bound = brightness_range["lower"]
            upper_bound = brightness_range["upper"]

            # Apply modifications to v_channel based on the adjustment value and brightness range
            # For example:
            v_channel[(v_channel >= lower_bound) & (v_channel <= upper_bound)] += adjustment

        # Clip values to stay within the valid range (0 to 255)
        v_channel = np.clip(v_channel, 0, 255)

        # Convert back to uint8
        v_channel = v_channel.astype(np.uint8)

        # Merge the modified V channel back into the HSV image
        modified_hsv = cv2.merge([h_channel, s_channel, v_channel])

        # Convert the modified HSV image back to BGR for display
        processed_image = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

        # # Display the original and modified images
        # cv2.imshow('Original Image', self.display_image.image)
        # cv2.imshow('Modified Image', processed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return processed_image   
























           



    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
                # # Print the histogram BEFORE modification
                # print("Histogram BEFORE modification:")
                # for i, bin_ in enumerate(bins[:-1]):
                #     print(f'Bin {i+1} ({bin_} - {bins[i+1]}): {hist_before[i]}')

                # # Plot the histogram BEFORE modification
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.hist(v.flatten(), bins=15, range=[0, 256], color='b', rwidth=0.8, label='Before Modification')
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                # plt.title('Histogram of V (Value) Channel BEFORE and AFTER Modification')

                # # Loop through each bin and apply the specified brightness adjustment
                # for bin_idx, amount_to_add in enumerate(brightness_values):
                #     if amount_to_add != 0:
                #         v[(v >= bins[bin_idx]) & (v < bins[bin_idx + 1])] += amount_to_add

                # # Calculate the histogram of the V (value) channel AFTER modification
                # hist_after, _ = np.histogram(v.flatten(), 15, [0, 255])

                # # Print the histogram AFTER modification
                # print("\nHistogram AFTER modification:")
                # for i, bin_ in enumerate(bins[:-1]):
                #     print(f'Bin {i+1} ({bin_} - {bins[i+1]}): {hist_after[i]}')

                # # Plot the histogram AFTER modification
                # plt.hist(v.flatten(), bins=15, range=[0, 256], color='r', rwidth=0.8, alpha=0.7, label='After Modification')
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                # plt.legend()
                # plt.title('Histogram of V (Value) Channel BEFORE and AFTER Modification')

                # # Show the two histograms
                # plt.tight_layout()
                # plt.show()




                # Display the individual channels
                # cv2.imshow('Hue', h)
                # cv2.imshow('Saturation', s)
                # cv2.imshow('Value', v)
                # cv2.waitKey(0)




















                # # Create a histogram of the v channel
                # hist = np.histogram(v, bins=15, range=(0, 256))

                # # Print the pixel count for each bin
                # for i in range(len(hist[0])):
                #     print(hist[0][i])

                # # Create a histogram of the v channel
                # hist = np.histogram(v, bins=15, range=(0, 256))
                # print(f'Histogram: {hist}')

                # # Plot the histogram
                # plt.hist(hist[0])

                # # Set the title and labels
                # plt.title('Histogram of V Channel')
                # plt.xlabel('Pixel Value')
                # plt.ylabel('Pixel Count')

                # # Show the plot
                # plt.show()












     
                # # Print the h, s, and v values for each pixel
                # for i in range(h.shape[0]):
                #     for j in range(h.shape[1]):
                #         print(h[i, j], s[i, j], v[i, j])




                # # Display the individual channels
                # cv2.imshow('Hue', h)
                # cv2.imshow('Saturation', s)
                # cv2.imshow('Value', v)
                # cv2.waitKey(0)













    def shift_channel(channel, value):
        # Shift the channel values by the given value, but clip at 0 and 255
        new_channel = np.clip(channel.astype(np.int32) + value, 0, 255).astype(np.uint8)
        return new_channel


    def key_handler(self, key):
        change_classification = False
        should_terminate = False

        # Check if the key is in the key_labels dictionary
        if key in self.key_labels:
            print(f"Key Pressed: {self.key_labels[key]}")
            # Handle the key based on its label
            label = self.key_labels[key]

            if label == 'Escape':
                should_terminate = True
            elif label in ['Good', 'Bad', 'Aesthetics', 'Duplicate', 'Unclassified']:
                change_classification = True
            elif label in ['Alt-v']:
                if ImageProcessor.show_sharpened_image == True:
                    ImageProcessor.show_sharpened_image == False
                else:
                    ImageProcessor.show_sharpened_image = True
 
            # print(f' I am returning {change_classification},{should_terminate},{label}')
            return change_classification, should_terminate, label

    def cv2PIL(self):
      
        # if self.image_display_type == 'Grid': 
        rgb_image = cv2.cvtColor(self.display_image.image_new, cv2.COLOR_BGR2RGB)
        # else:
        #     rgb_image = cv2.cvtColor(self.display_image.image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)

        self.tk_image = ImageTk.PhotoImage(pil_image)

    def load_from_dataframe(self, row):
        self.image_id = row['Image_ID']
        self.fname = row['File_Name']
        print(f'image_id {self.image_id}')        

        self.orientation = None  # Assuming you don't have 'Orientation' in your dataframe
        self.image = np.frombuffer(row['Original_Image'], dtype=np.uint8).reshape(1344, 2016, 3)
        self.original_image_height = row['Original_Height']
        self.original_image_width = row['Original_Width']
        self.scale_factor = (row['Scale_Up'], row['Scale_Down'])

        self.orientation = self.get_orientation()
        self.get_exif_data()
        self.get_classification()
        self.image_gs = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.image_dnz = cv2.fastNlMeansDenoisingColored(self.image.astype(np.uint8), None, 2, 2, 7, 21)
        self.image_sharp = self.sharpen()
        self.get_brightness()
        self.get_contrast()
        self.haze_factor = self.brightness / self.contrast
        self.get_hough_lines()

    
    def initial_display(self, filter_set = False):
        # Ensure that the dataset is loaded
        







        self.image_processor.load_data_from_data_handler(ImageProcessingApp.sql_path,ImageProcessingApp.table_sheet, self.source_var.get(), self.sort_var.get())



        # Valid Files in Parent Directory
        valid_files = ImageProcessingApp.get_valid_original_files()
        print(f'Return from initial display Valid Files\n{valid_files}')


        self.file_stats["Images"] = self.verify_dataframe_images_with_originals(valid_files)     
        filing_df = self.recalculate_file_stats(self.image_processor.images_pd)
        # print(f'Name, Classification df {filing_df}')
        if not filing_df.empty:
            response = messagebox.askyesno("Print to Console", "Reset Directories?")
            if response:
                self.reset_classified_files(filing_df)
                self.recalculate_file_stats(self.image_processor.images_pd)

        # Initialize the flags for the image
        change_classification = False 
        should_terminate = False
        display_image = None

        self.image_display_type = 'Grid'
        self.sorted_record_index = 0
        self.num_images = len(self.image_processor.sorted_pd)

        print(f' Head: {self.image_processor.sorted_pd.head()}')
        row = self.image_processor.sorted_pd.iloc[self.sorted_record_index]
        print(f'row: {row}')
        image_name = row['File_Name']
        image_id = row['Image_ID']
        orientation = row['Orientation']
        original_index = row['Original Index']
        original_height = row['Original_Height']
        original_width = row['Original_Width']
        scale_factor = (row['Scale_Down'],row['Scale_Up'])

        # Convert the binary data to a NumPy array
        display_array = np.frombuffer(row['Sharpened_Image'], np.uint8)

        # Decode the image using OpenCV
        display_image = cv2.imdecode(display_array, cv2.IMREAD_COLOR)

        # cv2.imshow("Display Image", display_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        
        

        print(f'image_id, image_name, scale_factor[0], original_height, original_width {image_id}, {image_name}, {scale_factor[0]}, {original_height}, {original_width}')
        self.display_image = ColorImage(display_image, image_id, image_name, scale_factor[0], original_height, original_width)

        # Convert OpenCV image to PIL (tk) image
        self.cv2PIL()

        # Add the image to the Canvas
        self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        self.update_stats()
        self.initial_image_flag = False
        self.root.menu_frame.button_frame.display_data_btn.state = 'Normal'
        

        # Calculate the histogram
        hist = cv2.calcHist([display_image], [0], None, [256], [0, 256])

        # Plot the histogram
        plt.figure()
        plt.plot(hist, color='black')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Brightness Histogram')
        plt.grid(True)
        plt.show()

        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)



        # for j in range(1):
        #     # Apply a binary threshold to the grayscale image
        #     strt_pt = 100
        #     _, binary_image = cv2.threshold(gray_image, strt_pt  + j*25, 255, cv2.THRESH_BINARY)

        #     # Use the binary image as a mask to isolate the corresponding regions in the original image
        #     masked_image = cv2.bitwise_and(display_image, display_image, mask=binary_image)

        #     # Display the masked image
        #     cv2.imshow('Masked Image ' + str(j), masked_image)
        #     cv2.imshow('Binary Image ' + str(j), binary_image)
            
        #     threshold_adaptive = cv2.adaptiveThreshold(
        #         gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #     AT_image = cv2.bitwise_and(display_image, display_image, mask=threshold_adaptive)
            
        #     cv2.imshow('Adaptive Threshold' + str(j), threshold_adaptive)
        #     cv2.imshow('AT Image' + str(j), AT_image)

            

        #     _, threshold_otsu = cv2.threshold(
        #         gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #     otsu_image = cv2.bitwise_and(display_image, display_image, mask=threshold_otsu)

        #     cv2.imshow('OTSU' + str(j), threshold_otsu)
        #     cv2.imshow('OTSU Image' + str(j), otsu_image)


        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




        # # Find contours in the binary image
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Find the largest contour (dominant object)
        # largest_contour = max(contours, key=cv2.contourArea)

        # # Get the bounding box coordinates for the largest contour
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # print(f'x,y,w,h: {x}, {y}, {w}, {h}')

        # # Draw the bounding box on the original image
        # cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Display the original image with the bounding box
        # cv2.imshow('Binary Image', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        





        # # Display the original image with the bounding box
        # cv2.imshow('Dominant Object Bounding Box', display_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()









        '''       
        num_levels = 15
        brightness_ranges = create_brightness_ranges(num_levels)
        hue_ranges = create_hue_ranges(num_levels)
        saturation_ranges = create_saturation_ranges(num_levels)
        s_v_ranges = create_s_v_ranges(num_levels)
        
        hsv = cv2.cvtColor(display_image, cv2.COLOR_BGR2HSV)
        h_channel,s_channel, v_channel  = cv2.split(hsv)


        brightness_masks = create_brightness_masks(brightness_ranges, v_channel)
        hue_masks = create_hue_masks(hue_ranges, h_channel)
        saturation_masks  = create_saturation_masks(saturation_ranges, s_channel)
        s_v_masks  = create_s_v_masks(saturation_masks, brightness_masks, num_levels)
        inverse_s_v_masks  = create_inverse_s_v_masks(saturation_masks, brightness_masks, num_levels)

        for y in range(num_levels):
            b_mask = brightness_masks[f"brightness_{y+1}"] 
            b_result = cv2.bitwise_and(display_image, display_image, mask=b_mask)

            s_mask = saturation_masks[f"saturation_{y+1}"] 
            s_result = cv2.bitwise_and(display_image, display_image, mask=s_mask)

            h_mask = hue_masks[f"hue_{y+1}"] 
            h_result = cv2.bitwise_and(display_image, display_image, mask=h_mask)

            s_v_mask = s_v_masks[f"s_v_{y+1}"] 
            s_v_result = cv2.bitwise_and(display_image, display_image, mask=s_v_mask)

            inverse_s_v_mask = inverse_s_v_masks[f"inverse_s_v_{y+1}"] 
            inverse_s_v_result = cv2.bitwise_and(display_image, display_image, mask=inverse_s_v_mask)


            display_masks(b_result, s_result, h_result, s_v_result,inverse_s_v_result,y+1)



            # Wait for a key press and close the windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # cv2.imshow(f"s_v_{y+1}", result)
            # cv2.imshow(f"hue_{y+1}", result)
            # cv2.imshow(f"saturation_{y+1}", result)
            # cv2.imshow(f"brightness_{y+1}", result)












        # # # Apply the brightness masks to the original image
        # # for key, mask in brightness_masks.items():

        # # # Apply the hue masks to the original image
        # # for key, mask in hue_masks.items():
        # #     result = cv2.bitwise_and(display_image, display_image, mask=mask)
        # #     cv2.imshow(key, result)

        # # # Apply the saturation masks to the original image
        # # for key, mask in saturation_masks.items():
        # #     result = cv2.bitwise_and(display_image, display_image, mask=mask)
        # #     cv2.imshow(key, result)

        # # Apply the s_v masks to the original image
        # for key, mask in s_v_masks.items():
        #     result = cv2.bitwise_and(display_image, display_image, mask=mask)
        #     cv2.imshow(key, result)


        '''


    def verify_dataframe_images_with_originals(self, valid_files):
        found_images = 0
        for file in valid_files:
            file_name =  f'{self.dir_path}/{file}'
            print(f'verify_dataframe_images_with_originals: {file_name}')
            print(f'Number of images in self.images_pd {len(self.image_processor.images_pd)}')
            print(f'Number of Valid files {len(valid_files)}')


            try:
                row = self.image_processor.images_pd.loc[self.image_processor.images_pd['File_Name'] == file_name]
                print(f'Trying: {file_name}')
                if not row.empty:
                    found_images += 1
            except:
                print(f'')
                pass
                print(f'File {file_name} has no companion image')        
        print(f'Valid Files:  {len(valid_files)}')
        print(f'Found Images: {found_images}')

        # if len(valid_files) == found_images:               
        return found_images


    def update_display(self):

        row = self.image_processor.sorted_pd.iloc[self.sorted_record_index]
        print(f'row: {row}')
        image_name = row['File_Name']
        image_id = row['Image_ID']
        orientation = row['Orientation']
        original_index = row['Original Index']
        original_height = row['Original_Height']
        original_width = row['Original_Width']
        scale_factor = (row['Scale_Down'],row['Scale_Up'])

        # Convert the binary data to a NumPy array
        display_array = np.frombuffer(row['Sharpened_Image'], np.uint8)

        # Decode the image using OpenCV
        display_image = cv2.imdecode(display_array, cv2.IMREAD_COLOR)

        # cv2.imshow("Display Image", display_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        
        print(f'image_id, image_name, scale_factor[0], original_height, original_width {image_id}, {image_name}, {scale_factor[0]}, {original_height}, {original_width}')

        self.display_image = ColorImage(display_image, image_id, image_name, scale_factor[0], original_height, original_width)


        # Convert image to useable tk image
        self.cv2PIL()

        # Add the image to the Canvas
        self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        self.update_stats()


        # # Calculate the histogram
        # hist = cv2.calcHist([display_image], [0], None, [256], [0, 256])

        # # Plot the histogram
        # plt.figure()
        # plt.plot(hist, color='black')
        # plt.xlabel('Pixel Intensity')
        # plt.ylabel('Frequency')
        # plt.title('Brightness Histogram')
        # plt.grid(True)
        # plt.show()

        # # Convert the image to grayscale
        # gray_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)

        # # Apply a binary threshold to the grayscale image
        # _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)




        # # Use the binary image as a mask to isolate the corresponding regions in the original image
        # masked_image = cv2.bitwise_and(display_image, display_image, mask=binary_image)

        # # Display the masked image
        # cv2.imshow('Masked Image', masked_image)
        # cv2.imshow('Binary Image', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




        # # Find contours in the binary image
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # Find the largest contour (dominant object)
        # largest_contour = max(contours, key=cv2.contourArea)

        # # Get the bounding box coordinates for the largest contour
        # x, y, w, h = cv2.boundingRect(largest_contour)
        # print(f'x,y,w,h: {x}, {y}, {w}, {h}')

        # # Draw the bounding box on the original image
        # cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # # Display the original image with the bounding box
        # cv2.imshow('Binary Image', binary_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
  















        # num_levels = 15
        # brightness_ranges = create_brightness_ranges(num_levels)
        # hue_ranges = create_hue_ranges(num_levels)
        # saturation_ranges = create_saturation_ranges(num_levels)
        # s_v_ranges = create_s_v_ranges(num_levels)
        
        # hsv = cv2.cvtColor(display_image, cv2.COLOR_BGR2HSV)
        # h_channel,s_channel, v_channel  = cv2.split(hsv)


        # brightness_masks = create_brightness_masks(brightness_ranges, v_channel)
        # hue_masks = create_hue_masks(hue_ranges, h_channel)
        # saturation_masks  = create_saturation_masks(saturation_ranges, s_channel)
        # s_v_masks  = create_s_v_masks(saturation_masks, brightness_masks, num_levels)
        # inverse_s_v_masks  = create_inverse_s_v_masks(saturation_masks, brightness_masks, num_levels)

        # for y in range(num_levels):
        #     b_mask = brightness_masks[f"brightness_{y+1}"] 
        #     b_result = cv2.bitwise_and(display_image, display_image, mask=b_mask)

        #     s_mask = saturation_masks[f"saturation_{y+1}"] 
        #     s_result = cv2.bitwise_and(display_image, display_image, mask=s_mask)

        #     h_mask = hue_masks[f"hue_{y+1}"] 
        #     h_result = cv2.bitwise_and(display_image, display_image, mask=h_mask)

        #     s_v_mask = s_v_masks[f"s_v_{y+1}"] 
        #     s_v_result = cv2.bitwise_and(display_image, display_image, mask=s_v_mask)

        #     inverse_s_v_mask = inverse_s_v_masks[f"inverse_s_v_{y+1}"] 
        #     inverse_s_v_result = cv2.bitwise_and(display_image, display_image, mask=inverse_s_v_mask)


        #     display_masks(b_result, s_result, h_result, s_v_result,inverse_s_v_result,y+1)
        
        # # Wait for a key press and close the windows
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()







    def accept_crop(self):
        
        image_name = self.display.display_image_file_name
        current_original_index = self.display.display_original_index
        image_id = self.display.display_image_id
        orientation = self.display.display_image_orientation

        # load image then resize then display....two steps
        image = self.load_image(image_name)


        # Coordinates of the rectangle on the displayed landscape image
        crop_display_coordinates = self.root.image_frame.canvas.coords(self.rectangle_id)

        # Map to original_coordinates        
        original_crop_coordinates = (int(crop_display_coordinates[0] * (1/.3)), int(crop_display_coordinates[1]  * (1/.3)),\
         int(crop_display_coordinates[2]  * (1/.3)), int(crop_display_coordinates[3]  * (1/.3)))




        # self.segment = Segment(whole image,orientation, image_id, file name, original_shape, original_crop_coordinates )
        self.segment = Segment(orientation, image_id, image_name, original_crop_coordinates)

        t = Color_Image(self.segment)
        sharp_file = t.image_sharp


        # Specify the directory to save the sharpened image
        save_directory = 'D:/Cropped Images/'

        # Save the sharpened image
        file_name = os.path.basename(self.display.display_image_file_name)

        cv2.imwrite(f'{save_directory}{file_name}', t.image_sharp)

        self.create_display_image(image, orientation, image_id, image_name, original_index)







        # Get file name of the displayed image and load original (we just accessed the original to create the display_image)

        # file_name = self.image_processor.sorted_pd.at[self.sorted_record_index, 'File_Name']



        




        # Returns segment of original
        


        # Translate back to display

        # Get shape from the display_image
        display_landscape_height = self.display.display_image_height    # (actuallly it is 1344)
        display_landscape_width = self.display.display_image_width      # (actually it is 2016)

        # Calculate scaling factors to original (We can do this because image ratio is constant.
        # Would also work for square images)
        
        scaling_factor =  original_image_segment.shape[1]/self.display.display_image_width


        # Find the original_index of the current record in sorted_pd
        # Set ROI_coordinates for the corresponding record in images_pd
        
        if 'ROI_coordinates' not in self.image_processor.images_pd.columns:
             
            # Create a new column 'ROI_coordinates' in images_pd
            self.image_processor.images_pd['ROI_coordinates'] = pd.Series([None] * len(self.image_processor.images_pd))

        # Set ROI_coordinates for the corresponding record in images_pd
        self.image_processor.images_pd.loc[current_original_index, 'ROI_coordinates'] = f'({coordinate_translator[0]},\
        {coordinate_translator[1]},{coordinate_translator[2]},{coordinate_translator[3]})'


        # Assuming you have updated the ROI_coordinates as discussed
        print("Original Image ID:", self.image_processor.images_pd.at[index, 'Image_ID'])
        print("Updated ROI_coordinates:", self.image_processor.images_pd.at[index, 'ROI_coordinates'])


        








        # cv2.imshow("Sharpened Image Segment", t.image_sharp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




        

        print(f'Original Image Shape')
        print(f'Original Landscape Height = {original_landscape_height}')
        print(f'Original Landscape Width = {original_landscape_width}')
        print(f'\nDisplay Shape')
        print(f'Display Landscape Height = {display_landscape_height}')
        print(f'Display Landscape Width = {display_landscape_width}')
        print(f'\nCropped Display')
        print(f'Rectangle ID: {self.rectangle_id}')
        print(f'Crop Coordinates:  {crop_display_coordinates[0]},{crop_display_coordinates[1]} {crop_display_coordinates[2]}, {crop_display_coordinates[3]}')
        print(f'Crop Display Height = {crop_display_height}')
        print(f'Crop Display Width = {crop_display_width}')
        print(f'\nOriginal_Image Segment')
        print(f'Original Image Segment Coordinates  = {coordinate_translator[2]},{coordinate_translator[3]} {coordinate_translator[0]},{coordinate_translator[1]}')
        print(f'Original Image Segment Shape: {original_image_segment.shape[0]} x {original_image_segment.shape[1]}')
        print(f'\nScaling factor to determine coordinates to display: {scaling_factor}')
        print(f'\nResized Sharpened Dimensions : {resized.shape}')

















        # row = self.image_processor.sorted_pd.iloc[self.sorted_record_index]
        # # image_id = row['Image_ID']
        # # print(f'Row: {row}')
        # image, orientation, image_id,file_name, original_shape = self.resize_file_for_display(self.sorted_record_index)
        # self.display = DisplayImage( image, orientation, image_id, file_name, original_shape)
        
        # t = Color_Image(self.display.display_image)
        # self.display.display_image = t.image_sharp

        # if self.image_display_type == 'Grid':
        #     if self.display.display_image_orientation == 'Landscape':
        #         self.display.add_grid(self.landscape_grid)
        #     else:
        #         self.display.add_grid(self.portrait_grid)

        # # Convert image to useable tk image
        # self.cv2PIL()

        # # Add the image to the Canvas
        # self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        # self.initial_image_flag = False
        





        # # Convert image to useable tk image
        # self.cv2PIL()

        # # Add the image to the Canvas
        # self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        

        # self.update_stats()

    def on_right_arrow_key(self, event = None):

        new_current_row = self.sorted_record_index + 1
        if new_current_row == self.num_images:
            return
        self.sorted_record_index +=1
        # print(f'new_current_row: {self.sorted_record_index}')
        self.update_display()

    def is_point_inside_rectangle(self, x,y):
        if self.rectangle_id !=0:
            print(f'Rectangle ID: {self.rectangle_id} Point x: {x} Point y: {y}')
            print(f'Rectangle Coordinates: {self.root.image_frame.canvas.coords(self.rectangle_id)}')
            rectangle_coords = self.root.image_frame.canvas.coords(self.rectangle_id)
            x1,y1,x2,y2 = rectangle_coords

            # enclosed_items = self.root.image_frame.canvas.find_enclosed(x1, y1, x2, y2)
            if x > x1 and x < x2 and y > y1 and y < y2:
                self.move_operation = True
                return
        self.move_operation = False       

            # print(f'Enclosed Items {len(enclosed_items)}')
            # if len(enclosed_items) > 0:

    def calculate_aspect_ratio(self, start_x, start_y, end_x, end_y):
        # Calculate the aspect ratio
        aspect_ratio = 1.5  # Change this to the desired aspect ratio (width/height)

        # Calculate the width and height of the original rectangle
        width = end_x - start_x
        height = end_y - start_y

        # Adjust the width or height to maintain the aspect ratio
        if width > height:
            new_height = width / aspect_ratio
            new_end_y = start_y + new_height
            return end_x, new_end_y
        else:
            new_width = height * aspect_ratio
            new_end_x = start_x + new_width
            return new_end_x, end_y

    def print_crop_data(self):
        print(f'Original Image Shape')
        print(f'Original Landscape Coordinates')
        print(f'Original Aspect Ratio')
        print(f'Original Landscape Height = {original_landscape_height}')
        print(f'Original Landscape Width = {original_landscape_width}')

        print(f'\nDisplay Shape')
        print(f'Original Image Shape')
        print(f'Display Landscape Coordinates')
        print(f'Display Aspect Ratio')
        print(f'Display Landscape Height = {display_landscape_height}')
        print(f'Display Landscape Width = {display_landscape_width}')
        print(f'\nCropped Display')

        print(f'Rectangle ID: {self.rectangle_id}')
        print(f'Crop Image Shape')
        print(f'Crop Coordinates:  {crop_display_coordinates[0]},{crop_display_coordinates[1]} {crop_display_coordinates[2]}, {crop_display_coordinates[3]}')
        print(f'Crop Aspect Ratio')
        print(f'Crop Display Height = {crop_display_height}')
        print(f'Crop Display Width = {crop_display_width}')

        print(f'\nOriginal_Image Segment')
        print(f'Original Image Segment Shape: {original_image_segment.shape[0]} x {original_image_segment.shape[1]}')
        print(f'Original Image Segment Coordinates  = {coordinate_translator[2]},{coordinate_translator[3]} {coordinate_translator[0]},{coordinate_translator[1]}')
        print(f'\nScaling factor to determine coordinates to display: {scaling_factor}')

        print(f'\nResized Sharpened Dimensions : {resized.shape}')




        # cv2.imshow("Resized Sharpened Image", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def on_press(self, event):
        # Store the start coordinates
        self.start_x = event.x
        self.start_y = event.y
        print(f'On Press: {self.start_x},{self.start_y} ')

        # rectangle_id = 1  # Replace with the actual rectangle_id
        point_x = event.x       # Replace with the x-coordinate of the point
        point_y = event.y     # Replace with the y-coordinate of the point
        
        print(f'point_x: {point_x} point_y: {point_y} event.x: {event.x} event.y: {event.y}')
        self.is_point_inside_rectangle(point_x, point_y)
        
        if self.move_operation:
            print("Point is inside the rectangle")
            self.move_operation = True
            self.move_start_x = point_x
            self.move_start_y = point_y
        else:
        # Store the start coordinates
            print("Point is outside the rectangle")
            self.move_operation = False
            self.start_x = event.x
            self.start_y = event.y

    def on_release(self, event):
            self.start_x = 0
            self.start_y = 0
            self.move_operation = False

    def on_drag(self, event):
        # print(f'On Entry into drag: {self.start_x},{self.start_y} ')
        # print(f'{self.move_operation}')
        current_x = self.root.image_frame.canvas.canvasx(event.x)
        current_y = self.root.image_frame.canvas.canvasy(event.y)

        if self.move_operation:
            # Move the existing rectangle during drag
            move_x = current_x - self.move_start_x
            move_y = current_y - self.move_start_y
            self.root.image_frame.canvas.move(self.rectangle_id, move_x, move_y)
            self.move_start_x = current_x
            self.move_start_y = current_y        


        else:
            # print(f'Before Update in drag: {self.start_x},{self.start_y} {current_x},{current_y}')
            # Update the rectangle during drag
            # self.start_x, self.start_y = self.draw_rectangle(self.start_x,self.start_y,current_x, current_y)
            # print(f'After Update in drag: {self.start_x},{self.start_y} {current_x},{current_y}')

            # Calculate the new end coordinates based on the aspect ratio
            new_end_x, new_end_y = self.calculate_aspect_ratio(self.start_x, self.start_y, current_x, current_y)

            # Update the rectangle during drag
            self.start_x, self.start_y = self.draw_rectangle(self.start_x, self.start_y, new_end_x, new_end_y)

    def draw_rectangle(self, sx, sy, end_x, end_y, ol = "black"):
        # print(f'Draw rect: {sx}, {sy} {end_x},{end_y}')
        # Clear previous rectangles
        if ol == "black":
            self.root.image_frame.canvas.delete("rectangle")
        else:
            print(f'Outline: {ol}')
        # Draw the new rectangle
        self.rectangle_id =self.root.image_frame.canvas.create_rectangle(
            sx, sy, end_x, end_y, outline= ol, width = 4, tags="rectangle"
             )
      
        # print(f' Before leaving Draw rect: {sx}, {sy} {end_x},{end_y}')
        return sx, sy





        # def on_release(self, event):
        #     # Retrieve the release coordinates
        #     end_x = event.x
        #     end_y = event.y

        #     # Draw a rectangle with a black border
        #     self.root.image_frame.canvas.create_rectangle(
        #         self.start_x,
        #         self.start_y,
        #         end_x,
        #         end_y,
        #         outline="black",
        #         width=2
        #     )

        # Reset start coordinates to None for the next press
        self.start_x = None
        self.start_y = None

    def on_left_arrow_key(self, event = None):

        new_current_row = self.sorted_record_index - 1
        if new_current_row < 0:
            return
        self.sorted_record_index -=1
        self.update_display()

    def on_home_key(self, event):

        if self.sorted_record_index == 0:
            return
        self.sorted_record_index = 0
        self.update_display()

    def on_end_key(self, event):

        if self.sorted_record_index == self.num_images - 1:
            return
        self.sorted_record_index = self.num_images - 1
        self.update_display()

    def on_classification_key(self, event, keypress):

        change_classification_request_valid = False
        should_terminate = False

        # Check if the key is in the key_labels dictionary
        if keypress in self.key_labels:
            print(f"Key Pressed: {self.key_labels[keypress]}")
            # Handle the key based on its label
            label = self.key_labels[keypress]

            if label in ['Aesthetics', 'Bad', 'Duplicate', 'Good', 'Unclassified']:
                change_classification_request_valid = True

            if change_classification_request_valid and keypress != self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification']:
                print(f'Change_classification_request_valid {change_classification_request_valid}\nCurrent Classification: {self.image_processor.sorted_pd.at[self.sorted_record_index,"Classification"]}') 


                self.image_processor.previous_classification = self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification']
                self.image_processor.change_classification_request_success = self.classification_change(label)
                if self.image_processor.change_classification_request_success:
                    self.file_ops()

                self.image_processor.change_classification_request_success = False
                self.on_right_arrow_key()

    def classification_change(self,label):
        
        classification_change_success = False

        if label == 'Unclassified':
            self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification'] = 'U'
            self.record_index = self.image_processor.sorted_pd.loc[self.sorted_record_index, 'Original Index']
            self.image_processor.images_pd.at[self.record_index, 'Classification'] = 'U'
            classification_change_success = True

        elif label == 'Good':
            self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification'] = 'G'
            self.record_index = self.image_processor.sorted_pd.loc[self.sorted_record_index, 'Original Index']
            self.image_processor.images_pd.at[self.record_index, 'Classification'] = 'G'
            classification_change_success = True

        if label == 'Bad':
            self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification'] = 'B'
            self.record_index = self.image_processor.sorted_pd.loc[self.sorted_record_index, 'Original Index']
            self.image_processor.images_pd.at[self.record_index, 'Classification'] = 'B'
            classification_change_success = True

        if label == 'Aesthetics':
            self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification'] = 'A'
            self.record_index = self.image_processor.sorted_pd.loc[self.sorted_record_index, 'Original Index']
            self.image_processor.images_pd.at[self.record_index, 'Classification'] = 'A'
            classification_change_success = True

        if label == 'Duplicate':
            self.image_processor.sorted_pd.at[self.sorted_record_index, 'Classification'] = 'D'
            self.record_index = self.image_processor.sorted_pd.loc[self.sorted_record_index, 'Original Index']
            self.image_processor.images_pd.at[self.record_index, 'Classification'] = 'D'
            classification_change_success = True

        # print(f'Previous Classification: {self.image_processor.previous_classification}')        
  
        self.image_processor.create_final_rf_pd()
        return classification_change_success

    def delete_files(self,files_to_delete):

        for file_to_delete in files_to_delete:
            try:
                os.remove(file_to_delete)
                # print(f"File {file_to_delete} deleted successfully.")
            except Exception as e:
                print(f"Error: {e}")
        print(f" {files_to_delete} Files associated with {ImageProcessor.parent_dir} deleted successfully.")

        return # len(files_to_delete)

    def print_filing_stats(self,file_stats):
        for key, value in file_stats.items():
            print(f"{key}: {value}")
        # print(f'____________________________________________________________________________')
        # print(f'Stats = {self.image_processor.stats}')
        # print(f'Distances = {self.image_processor.distances_df}')
        # print(f'____________________________________________________________________________')

    def file_ops(self):
        print(f'I am inside file_ops() method')
        original_file_path = self.image_processor.images_pd.at[self.record_index, 'File_Name']
        file_name = os.path.basename(original_file_path)
        parent_directory = os.path.dirname(original_file_path)
        files_to_delete = []
        record_to_save = self.image_processor.images_pd.iloc[[self.record_index]]
        current_classification = self.image_processor.images_pd.at[self.record_index, 'Classification']
        print(f'PREVIOUS CLASSIFICATION: {self.image_processor.previous_classification}')
        print(f'CURRENT CLASSIFICATION: {current_classification} ' )

        classification_map = {
        'A': ImageProcessingApp.aesthetics_dir,
        'B': ImageProcessingApp.rejected_dir,
        'D': ImageProcessingApp.duplicate_dir,
        'G': ImageProcessingApp.accepted_dir

        }

        # Delete from Previous classification subdirectory
        if self.image_processor.previous_classification in ['A','B','D','G']:
            files_to_delete .append(f'{classification_map[self.image_processor.previous_classification]}/{self.image_processor.previous_classification}{ file_name}')
        print(f'File Name: {file_name}\nCurrent Classification: {current_classification}\nFiles to Delete: {files_to_delete}')
        
        if len(files_to_delete):
            self.delete_files(files_to_delete)

        # Add to Current classification subdirectory
        if current_classification in ['A','B','D','G']:
            print(f'Inside Save')
            target_directory = classification_map[current_classification]
            new_file_path = os.path.join(target_directory, current_classification + file_name)
            shutil.copy(original_file_path, new_file_path)


        # Save the updated record to the SQL database
        data_handler = DataHandler()
        data_handler.save_record_to_sql(record_to_save, self.sql_path, self.table_sheet)

        ImageProcessingApp.recalculate_file_stats(self.image_processor.images_pd)
        self.print_filing_stats(self.file_stats)

    def rf_train(self):
 
        features, target = self.random_forest_model_instance.get_features_targets(self.image_processor.final_rf_pd)
        self.random_forest_model_instance.random_forest_train(features, target)

    def rf_test(self):
        if self.random_forest_model_instance:
            features, target = self.random_forest_model_instance.get_features_targets(self.image_processor.final_rf_pd)
            self.random_forest_model_instance.random_forest_test(features, target)

    def view_good_images(self):
        self.filter_data("G")
        
        # Initialize the flags for the image
        change_classification = False 
        should_terminate = False
        display_image = None

        self.image_display_type = 'No Grid'
        self.sorted_record_index = 0
        self.num_images = len(self.image_processor.sorted_pd)

        print(f' Head: {self.image_processor.sorted_pd.head()}')
        row = self.image_processor.sorted_pd.iloc[self.sorted_record_index]
        print(f'row: {row}')
        image_name = row['File_Name']
        image_id = row['Image_ID']
        orientation = row['Orientation']
        original_index = row['Original Index']
        original_height = row['Original_Height']
        original_width = row['Original_Width']
        scale_factor = (row['Scale_Down'],row['Scale_Up'])

        # Convert the binary data to a NumPy array
        display_array = np.frombuffer(row['Sharpened_Image'], np.uint8)

        # Decode the image using OpenCV
        display_image = cv2.imdecode(display_array, cv2.IMREAD_COLOR)

        # cv2.imshow("Display Image", display_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()        
        self.display_image = ColorImage(display_image, image_id, image_name, scale_factor[0], original_height, original_width)

        if self.image_display_type == 'Grid':
            if self.display.display_image_orientation == 'Landscape':
                self.display.add_grid(self.landscape_grid)
            else:
                self.display.add_grid(self.portrait_grid)

        
        # Convert image to useable tk image
        self.cv2PIL()

        # Add the image to the Canvas
        self.root.image_frame.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.root.image_frame.canvas.grid(row=0, column=0, sticky="nsew")  # Adjust row and column as needed        
        # self.start_x = 0
        # self.start_y = 0
        # self.update_stats()
        # self.initial_image_flag = False

    def filter_data(self,classification=None):

        if classification is not None:
            # Filter based on classification
            filtered_data = self.image_processor.sorted_pd[self.image_processor.sorted_pd['Classification'] == classification]
        else:
            # No filter, use the original DataFrame
            filtered_data = self.image_processor.sorted_pd  # Keep the DataFrame as is

        # Reset the index if needed
        filtered_data.reset_index(drop=True, inplace=True)

        # Update self.sorted_pd with the filtered DataFrame
        self.image_processor.sorted_pd = filtered_data
        self.sorted_record_index = 0
        self.num_images = 0


        # def get_segment(self,x1,y1,x2,y2):


    # def load_image(file_name):
    #     # Load the file 
    #     image = cv2.imread(file_name)
    #     # image_height = image.shape[0]
    #     # image_width = image.shape[1]
    #     # print(f'Original Image Shape: {image_height}x{image_width}')
    #     return image

    # def resize_image_for_display(O_image):
    #     scale_factor  = self. get_scale(O_image)
        
    #     aspect_ratio = O_image.shape[0]/O_image.shape[1]
    #     # print(f'Aspect Ratio: {aspect_ratio}')

    #     RO_image_height = int(O_image.shape[0] * scale_factor)
    #     RO_image_width = int(O_image.shape[1] * scale_factor)          

    #     RO_image_dim = (RO_image_width, RO_image_height)
    #     RO_image= cv2.resize(O_image,RO_image_dim, interpolation = cv2.INTER_AREA).astype(np.uint8)

    #     return (RO_image,scale_factor)

    # def get_scale(O_image):
        
    #     if O_image.shape[1] < O_image.shape[0]:
    #         scale_factor = 896/ O_image.shape[1]
    #     elif O_image.shape[1] > O_image.shape[0]:
    #         scale_factor = 2016/ O_image.shape[1]
    #     else:
    #         scale_factor = 1

    #     return scale_factor


            # # Calculate the histogram of the V (value) channel
            # hist_before, bins = np.histogram(v.flatten(), 15, [0, 255])

            # # Print the histogram BEFORE modification
            # print("Histogram BEFORE modification:")
            # for i, bin_ in enumerate(bins[:-1]):
            #     print(f'Bin {i+1} ({bin_} - {bins[i+1]}): {hist_before[i]}')

            # # Plot the histogram BEFORE modification
            # plt.figure(figsize=(12, 6))

            # plt.subplot(1, 2, 1)
            # plt.plot(bins[:-1], hist_before, 'bo-', label='Before')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title('Histogram of V (Value) Channel')
            # plt.legend()

            # # Loop through each bin and apply the specified brightness adjustment


            # plt.tight_layout()
            # plt.show()
            







            # # Calculate the histogram of the V (value) channel AFTER modification
            # hist_modified, _ = np.histogram(v.flatten(), 15, [0, 255])

            # # Print the histogram AFTER modification
            # print("\nHistogram AFTER modification:")
            # for i, bin_ in enumerate(bins[:-1]):
            #     print(f'Bin {i+1} ({bin_} - {bins[i+1]}): {hist_modified[i]}')

            # # Plot the histogram AFTER modification
            # plt.subplot(1, 2, 2)
            # plt.plot(bins[:-1], hist_modified, 'ro-', label='After')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.title('Histogram of V (Value) Channel')
            # plt.legend()

            # # Show the two histograms




            # # Loop through each bin and apply the specified brightness adjustment
