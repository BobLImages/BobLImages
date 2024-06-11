import pandas as pd
from file_functions_2 import *
from tkinter import messagebox
from data_handler import *
import time 
from image_processing_app import *


class ImageProcessor:

    has_changed = False
    previous_classification =[]

    resized_image = None
    sharpened_resized_image = None
    sharpened_grid_image = None
    grid_image = None
    
    statistics_window = None
    sort_order = None
    show_sharpened_image = False

 
    # Column mapping dictionary (key: original column name, value: SQL-compliant column name)
    column_mapping = {
        'Original_Image': 'Original Image',
        'Grayscale_Image': 'Grayscale Image',
        'Denoised_Image': 'Denoised Image',
        # 'Dehazed_Image': 'Dehazed Image',
        'Sharpened_Image': 'Sharpened Image',
        # Add other column mappings here
        'Image_ID': 'Image ID',
        'File_Name': 'File Name',
        'Haze_Factor': 'Haze Factor',
        'Hough_Info':'Hough Info',
        'Harris_Corners':'Harris Corners',
        'Hough_Circles':'Hough Circles',
        'Contour_Info' :'Contour Info',        
        'F_Stop': 'F-stop',
        'Black_Pixels': 'Black Pixels',
        'Mid_tone_Pixels': 'Mid-tone Pixels',
        'White_Pixels':  'White Pixels',
        'Classification': 'Classification',
        'SHV': 'SHV',
        'Faces': 'Faces',
        'Bodies': "Bodies",
        'Eyes': 'Eyes',
        'ISO': 'ISO',
        'Exposure': 'Exposure',
        'Variance': 'Variance',
        'Orientation': 'Orientation',
        'Brightness':'Brightness',
        'Contrast': 'Contrast',
        'Laplacian': 'Laplacian',
        'Original_Height': 'Original Height',
        'Original_Width': 'Original Width',
        'Scale_Up': 'Scale Up',
        'Scale Down': 'Scale Down'

    }





    # Define the column names as class attributes
    column_names = [
        'Image_ID', 'File_Name', 'Orientation', 'Brightness', 'Contrast', 'Haze_Factor', 'Hough_Info',
        'Hough_Circles', 'Harris_Corners', 'Contour_Info', 'Laplacian', 'SHV', 'Variance',
        'Exposure', 'F_Stop', 'ISO', 'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels',
        'Faces', 'Eyes', 'Bodies', 'Focal_Length', 'Classification', 'Original_Image',
        'Grayscale_Image', 'Denoised_Image', 'Sharpened_Image', 'Original_Height', 'Original_Width', 'Scale_Up', 'Scale_Down'
         ]


    # @classmethod
    # def initialize(cls, dir_path):
        
    #     print(f'dir_path {dir_path}\n\n\n\n')
    #     parent_directory, data_file_name = os.path.split(dir_path)
    #     cls.parent_dir = parent_directory
        
    #     cls.excel_path = "d:/Image Data Files/" + data_file_name + '.xlsx'
    #     cls.sql_path = "d:/Image Data Files sql/" + data_file_name + '.db'
    #     cls.table_sheet = "Sheet_1"
    #     # cls.parent_dir = parent_directory
    #     cls.accepted_dir = os.path.join(parent_directory, 'Accept')
    #     cls.rejected_dir = os.path.join(parent_directory, 'Reject')
    #     cls.aesthetics_dir = os.path.join(parent_directory, 'Aesthetics')
    #     cls.duplicate_dir = os.path.join(parent_directory, 'Duplicate')
        
    #     os.makedirs(cls.accepted_dir, exist_ok=True)
    #     os.makedirs(cls.rejected_dir, exist_ok=True)
    #     os.makedirs(cls.aesthetics_dir, exist_ok=True)
    #     os.makedirs(cls.duplicate_dir, exist_ok=True)

    #     # Check whether xlsx and sql db exists
    #     cls.excel_exists = os.path.exists(cls.excel_path)
    #     cls.sql_exists = os.path.exists(cls.sql_path)
    #     cls.file_stats["Valid Files"] = len(cls.get_valid_original_files())
    #     cls.get_file_stats()


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
 

    #     # Repopulate classified directories based on images_pd DataFrame
    #     for index, row in filing_df.iterrows():
    #         classification = row['Classification']
    #         file_name = os.path.basename(row['File_Name'])
    #         destination_dir = None

    #         if classification == 'G':
    #             destination_dir = cls.accepted_dir
    #         elif classification == 'B':
    #             destination_dir = cls.rejected_dir
    #         elif classification == 'A':
    #             destination_dir = cls.aesthetics_dir
    #         elif classification == 'D':
    #             destination_dir = cls.duplicate_dir
    #         else:
    #             print(f"Invalid classification for file: {file_name}")
    #             continue

    #         source_file_path = os.path.join(cls.parent_dir, file_name)
    #         destination_file_path = os.path.join(destination_dir, f"{classification}{file_name}")

    #         try:
    #             shutil.copy2(source_file_path, destination_file_path)
    #         except Exception as e:
    #             print(f"Error copying file {file_name}: {e}")

    #     print(f"Classified files have been reset.")


    # @classmethod
    # def recalculate_file_stats(cls,images_pd):
        
    #     # print(f'Here***************************************************************************')
    #     # Files in subdirectories
    #     good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = ImageProcessingApp.get_subdir_files()        

    #     # Rows by classification in dataframe
    #     good_df_rows, bad_df_rows, aesthetics_df_rows, duplicate_df_rows, unclassified_df_rows = cls.get_rows_by_classification(images_pd)        

    #     directories = {
    #         "Good Directory": (good_df_rows, good_directory_files),
    #         "Bad Directory": (bad_df_rows, bad_directory_files),
    #         "Aesthetics Directory": (aesthetics_df_rows, aesthetics_directory_files),
    #         "Duplicate Directory": (duplicate_df_rows, duplicate_directory_files)
    #     }

    #     for directory, (df_rows, dir_files) in directories.items():
    #         print(directory,(len(df_rows),len(dir_files)))
    #         missing_files = len(df_rows) - len(dir_files)
    #         invalid_files = len(dir_files) - len(df_rows)
    #         cls.file_stats[f"Missing from {directory}"] = missing_files
    #         cls.file_stats[f"Invalid in {directory}"] = invalid_files
    #         if missing_files > 0 or invalid_files > 0:
    #             filing_df = images_pd[['File_Name', 'Classification']]
    #             return filing_df

    #         filing_df = pd.DataFrame()
    #         return filing_df



    # @classmethod
    # def get_file_stats(cls):
            
    #     good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files = ImageProcessingApp.get_subdir_files()


    # @classmethod
    # def get_rows_by_classification(cls,images_pd):
    #     class_counts = images_pd['Classification'].value_counts()
        
    #     ImageProcessor.file_stats["Number of Unclassified Dataframe Images"] = class_counts.get('U', 0)
    #     ImageProcessor.file_stats["Number of Good Dataframe Images"] = class_counts.get('G', 0)
    #     ImageProcessor.file_stats["Number of Bad Dataframe Images"] = class_counts.get('B', 0)
    #     ImageProcessor.file_stats["Number of Aesthetics Dataframe Images"] = class_counts.get('A', 0)
    #     ImageProcessor.file_stats["Number of Duplicate Dataframe Images"] = class_counts.get('D', 0)

    #     # Generate lists of file names for each row of the dataframe corrected for classification 
    #     good_df_rows = set(images_pd.loc[images_pd['Classification'] == 'G', 'File_Name'].apply(lambda x: 'G' + os.path.basename(x)))
    #     bad_df_rows = set(images_pd.loc[images_pd['Classification'] == 'B', 'File_Name'].apply(lambda x: 'B' + os.path.basename(x)))
    #     aesthetics_df_rows = set(images_pd.loc[images_pd['Classification'] == 'A', 'File_Name'].apply(lambda x: 'A' + os.path.basename(x)))
    #     duplicate_df_rows = set(images_pd.loc[images_pd['Classification'] == 'D', 'File_Name'].apply(lambda x: 'D' + os.path.basename(x)))
    #     unclassified_df_rows = set(images_pd.loc[images_pd['Classification'] == 'U', 'File_Name'].apply(lambda x: '' + os.path.basename(x)))

    #     return good_df_rows, bad_df_rows, aesthetics_df_rows,duplicate_df_rows, unclassified_df_rows


    # @classmethod
    # def get_valid_original_files(cls):

    #     valid_files = []
    #     # Step 1: List all files in the directory
    #     parent_directory_files = set(os.listdir(cls.parent_dir))
    #     # print(f'Parent Directory File: {parent_directory_files}')
    #     for file in parent_directory_files:
    #         if file.endswith('.JPG') and '$' not in file and 'r_' not in file:
    #             valid_files.append(file)
    #     cls.file_stats["Valid Files"] = len(valid_files)
    #     print(f'Number of Valid Files: {len(valid_files)}')        
    #     # verify_dataframe_images_with_originals(valid_files)        

    #     return valid_files

    # @classmethod
    # def get_subdir_files(cls):
    
    #     all_items = os.listdir(cls.accepted_dir)
    #     # Filter out directories and their files
    #     good_directory_files = [item for item in all_items if os.path.isfile(os.path.join(cls.accepted_dir, item))]
    #     # print(f' Good Files: {len(good_directory_files)} {good_directory_files}')

    #     bad_directory_files = set(os.listdir(cls.rejected_dir))
    #     aesthetics_directory_files = set(os.listdir(cls.aesthetics_dir))
    #     duplicate_directory_files = set(os.listdir(cls.duplicate_dir))
    #     cls.file_stats["Number of Good Directory Files"] =  len(good_directory_files)
    #     cls.file_stats["Number of Bad Directory Files"] = len(bad_directory_files)
    #     cls.file_stats["Number of Aesthetics Directory Files"] = len(aesthetics_directory_files)
    #     cls.file_stats["Number of Duplicate Directory Files"] = len(duplicate_directory_files)

    #     return good_directory_files, bad_directory_files, aesthetics_directory_files, duplicate_directory_files
    

    @classmethod
    def get_cumlative_stats(cls,images_pd, just_mean = True):
        ''' Specify the columns you want to calculate statistics for
        ('Contrast', 'Haze_Factor', 'Hough_Info',
            'Hough_Circles', 'Harris_Corners', 'SHV', 'Variance',
            'Black_Pixels', 'Mid_tone_Pixels', 'White_Pixels')
        '''
        selected_columns = [
            'Brightness', 'Contour_Info',
            'Laplacian', 
        ]

        if just_mean:
            # Calculate mean for time being
            cls.stats = images_pd[selected_columns].agg(['mean'])
            # Transpose the statistics DataFrame to have columns as rows
            cls.stats = cls.stats.transpose()
            # Rename the columns for clarity
            # cls.stats.columns = ['b_Mean','c_Mean','l_Mean']
        else:
            # Calculate mean, min, max, and std for the selected columns
            cls.stats = images_pd[selected_columns].agg(['mean', 'min', 'max', 'std'])
            # Transpose the statistics DataFrame to have columns as rows
            # Rename the columns for clarity
            cls.stats = cls.stats.transpose()
            cls.stats.columns = ['b_Mean', '_b_Min', 'b_Max', 'b_Std',
            'c_Mean', 'c_Min', 'c_Max', 'c_Std',
            'l_Mean', 'l_Min', 'l_Max', 'l_Std',]
        
        # print(f'cls. stats\n{cls.stats}')
        return cls.stats

    @classmethod
    def calculate_distances(cls, images_pd):
        mean_brightness = cls.stats.loc['Brightness', 'mean']
        mean_contour_info = cls.stats.loc['Contour_Info', 'mean']
        mean_laplacian = cls.stats.loc['Laplacian', 'mean']

        # Calculate distances from mean for each category
        images_pd['d_mean_b'] = abs(images_pd['Brightness'] - mean_brightness)
        images_pd['d_mean_c'] = abs(images_pd['Contour_Info'] - mean_contour_info)
        images_pd['d_mean_l'] = abs(images_pd['Laplacian'] - mean_laplacian)

        # Create the new DataFrame with distances and 'image_id'
        cls.distances_df = images_pd[['Image_ID', 'd_mean_b', 'd_mean_c', 'd_mean_l']]


    def __init__(self):
        # Initialize the DataFrame with the predefined column names
        self.images_pd = pd.DataFrame(columns = self.column_names)
        self.sorted_pd = None
        self.final_rf_pd = None
        self.rf = None
        self.X_train = None

    # def verify_dataframe_images_with_originals(self, valid_files):
    #     found_images = 0
    #     for file in valid_files:
    #         file_name =  f'{ImageProcessor.path}/{file}'
    #         try:
    #             row = self.images_pd.loc[self.images_pd['File_Name'] == file_name]
    #             found_images += 1
    #         except:
    #             pass
    #             # print(f'File {file_name} has no companion image')        
    #     if len(valid_files) == found_images:               
    #         # print(f'Valid Files:  {len(valid_files)}')
    #         # print(f'Found Images: {found_images}')
    #         return found_images


    def load_data_from_data_handler(self, sql_path, table_sheet, source = 'sql', sort_source= 'Image_Id (Original Order)', data_filter = None):
        print(f'Source: {source}')
        # Call the DataHandler to load data
        data_handler = DataHandler()
        if source == "excel":
            self.images_pd = data_handler.load_from_excel(ImageProcessor.original_directory,
             ImageProcessor.excel_path,ImageProcessor.table_sheet)

        elif source == "sql":
            
            print(f'  *******************************************************************************************************') 
            print(f'  ** SQL Path: {sql_path}\n  ** Table or Sheet Name: {table_sheet}   image_processor Line 351')
            print(f'  *******************************************************************************************************') 

            self.images_pd = data_handler.load_from_sql(sql_path, table_sheet)
    









        # Get Mean Statistics
        ImageProcessor.get_cumlative_stats(self.images_pd)
        ImageProcessor.calculate_distances(self.images_pd)
        


        self.create_sorted_pd(sort_source)
        self.create_final_rf_pd()

    






        # print(f'Data Source:\t\t{source}')
        # print(f'Sort Source:\t\t{sort_source}')
        # print(f'Original Rows:\t\t{len(self.images_pd)}')
        # print(f'Sorted Rows:\t\t{len(self.sorted_pd)}')

        return self.images_pd




    def create_final_rf_pd(self):
            rf_train_data = self.images_pd  # Get the DataFrame from the ImageProcessor instance
            # Load the training data from the provided Excel file or DataFrame
            if isinstance(rf_train_data, pd.DataFrame) == False:
                raise ValueError("Invalid input. Please provide a DataFrame or an Excel file name.")

            # Preprocess the data if necessary (e.g., encoding categorical features)

            # Define a list of classifications to remove (A, D, U)
            classifications_to_remove = ['A', 'D', 'U']

            # Remove rows with the specified classifications
            self.final_rf_pd = rf_train_data[~rf_train_data['Classification'].isin(classifications_to_remove)]
            # Using head() to print the first x rows
            # print(self.final_rf_pd.head(10))
            # print(self.final_rf_pd.columns)

    def create_sorted_pd(self,sort_source):

        if sort_source == 'Image_Id (Original Order)':
            self.sorted_pd = self.images_pd
        elif sort_source == 'Brightness':
            self.sorted_pd = self.images_pd.sort_values(by='Brightness')
        elif sort_source == 'Contours':
            self.sorted_pd = self.images_pd.sort_values(by='Contour_Info')
        elif sort_source == 'Laplacian':
            self.sorted_pd = self.images_pd.sort_values(by='Laplacian')
        elif sort_source == 'Classification (U)':
            self.sorted_pd = self.images_pd.sort_values(by='Classification', ascending = False)
        elif sort_source == 'Classification (B)':
            self.sorted_pd = self.images_pd.sort_values(by='Classification')
        elif sort_source == 'SHV':
            self.sorted_pd = self.images_pd.sort_values(by='SHV')
        elif sort_source == 'd_mean_b':
            self.sorted_pd = self.images_pd.sort_values(by='d_mean_b')
        elif sort_source == 'd_mean_c':
            self.sorted_pd = self.images_pd.sort_values(by='d_mean_c')
        elif sort_source == 'd_mean_l':
            self.sorted_pd = self.images_pd.sort_values(by='d_mean_l')
        elif sort_source == 'Orientation-P':
            self.sorted_pd = self.images_pd.sort_values(by='Orientation',ascending = False)
        elif sort_source == 'Orientation-L':
            self.sorted_pd = self.images_pd.sort_values(by='Orientation')

        
        print(f' {self.sorted_pd.head()}')


        # Add a new column 'original_index' to store the original index values
        self.sorted_pd['Original Index'] = self.sorted_pd.index

        # Reset the index
        self.sorted_pd.reset_index(drop=True, inplace=True)



    def save_data_from_data_handler(self,target, dir_path = None, excel_path = None, sql_path = None, table_sheet = None, confirm_req = True):

        

        # print(f'  *******************************************************************************************************') 
        # print(f'  ** excel - {ImageProcessingApp.dir_path}\n  ** {ImageProcessingApp.excel_path}\n  ** {ImageProcessingApp.sql_path}\n  ** {ImageProcessingApp.table_sheet}')
        # print(f'  *******************************************************************************************************') 

        print(f'  *******************************************************************************************************') 
        print(f'  ** {target}\n  ** Directory Path: {dir_path}\n  ** Excel Path: {excel_path}\n  ** SQL Path: {sql_path}\n  ** Table or Sheet Name: {table_sheet}')
        print(f'  *******************************************************************************************************') 


        data_handler = DataHandler()
        if target == "excel":
            # print(f'Excel Path: {self.excel_path}')
            data_handler.save_data_to_excel(self.images_pd, excel_path, table_sheet, confirm_req)
        elif target == "sql":
            data_handler.save_data_to_sql(self.images_pd, sql_path, table_sheet, confirm_req)
        else:
            # Handle an invalid source if needed
            self.images_pd = None

        return self.images_pd

   
    def open_file_for_display(self,index):
        # Get the file path, image_id, and image orientation for the specified index
        file_path = self.sorted_pd.at[index, 'File_Name']
        
        # print(f'File Path: {file_path}')
        image_orientation = self.sorted_pd.at[index, 'Orientation']
        image_id = self.sorted_pd.at[index, 'Image_ID']
        image = cv2.imread(file_path)









    def prepare_image_for_display(self, index):

        # Get the file path, image_id, and image orientation for the specified index
        file_path = self.sorted_pd.at[index, 'File_Name']
        
        # print(f'File Path: {file_path}')
        image_orientation = self.sorted_pd.at[index, 'Orientation']
        image_id = self.sorted_pd.at[index, 'Image_ID']

        # rows = 5
        # columns = 7

        # # Load the file 
        image = cv2.imread(file_path)
        print(image.shape)
        # Resize the image using the resize_image function and place the grid on the resized image
        
        if image_orientation == 'Landscape':
            ImageProcessor.resized_image, p = resize_image(image, scale_percent=30)  # Adjust the scale percent as desired
            t = Color_Image(ImageProcessor.resized_image)
            ImageProcessor.sharpened_resized_image = t.image_sharp
            
        elif image_orientation == 'Portrait':
            ImageProcessor.resized_image, p = resize_image(image, scale_percent=20)  # Adjust thresized_image, p = resize_image(image, scale_percent=20)  # Adjust the scale percent as desired
            t = Color_Image(ImageProcessor.resized_image)
            ImageProcessor.sharpened_resized_image = t.image_sharp








        #     landscape_data = create_grids(1344, 2018, rows, columns)
        #     NW = landscape_data[0]
        #     zone_width = landscape_data[1]
        #     zone_height = landscape_data[2]
        #     print(f'NW: {NW}\nzone_width {zone_width}\nzone_height {zone_height}')


        #     portrait_data = create_grids(1344, 896, columns, rows)
        #     NW = portrait_data[0]
        #     zone_width = portrait_data[1]
        #     zone_height = portrait_data[2]
        
        # for nw_point in NW:
        #     # Calculate the coordinates of the top-left and bottom-right corners of the rectangle
        #     rectangle_start = nw_point
        #     rectangle_end = (nw_point[0] + zone_height, nw_point[1] + zone_width)
        #     cv2.rectangle(ImageProcessor.resized_image, rectangle_start, rectangle_end, (255, 255, 255), 2)
        #     cv2.rectangle(ImageProcessor.sharpened_resized_image , rectangle_start, rectangle_end, (255, 255, 255), 2)
        # ImageProcessor.grid_image = cv2.convertScaleAbs(ImageProcessor.resized_image)
        # ImageProcessor.sharpened_grid_image = cv2.convertScaleAbs(local_sharpened_resized_image)


    def create_plots(self, x_attributes,y_attributes):
        num_plots = len(x_attributes)
        num_rows = 2  # Number of rows for subplots
        num_cols = num_plots // num_rows  # Number of columns for subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

        # Define color mapping for classifications
        color_map = {'G': 'green', 'B': 'red', 'U': 'blue', 'A': 'cyan', 'D': 'yellow'}

        for idx, attribute in enumerate(x_attributes):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            # Get the classifications for this attribute (assuming it's in a column named 'Classification')
            classifications = self.images_pd['Classification']

            # Map the classifications to colors using the color_map
            colors = classifications.map(color_map)

            ax.scatter(self.images_pd[x_attributes[idx]], self.images_pd[y_attributes[idx]], c=colors)
            ax.set_xlabel(x_attributes[idx])
            ax.set_ylabel(y_attributes[idx])
            ax.set_title(x_attributes[idx] + '_' + y_attributes[idx])

        # Adjust layout for subplots
        plt.tight_layout()

        # Show the plots
        plt.show()


    def create_plots_2(self, x_attributes,y_attributes):
        num_plots = len(x_attributes)
        num_rows = 1  # Number of rows for subplots
        num_cols = num_plots  # Number of columns for subplots

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

        # Ensure axes is a 2-dimensional array even if there's only one plot
        if num_plots == 1:
            axes = [axes]


        # Define color mapping for classifications
        color_map = {'G': 'green', 'B': 'red', 'U': 'blue', 'A': 'cyan', 'D': 'yellow'}


        merged_df = pd.merge(ImageProcessor.distances_df, self.images_pd[['Image_ID', 'Classification']], on='Image_ID', how='left')

        # Get the classifications for this attribute (assuming it's in a column named 'Classification')
        classifications = merged_df['Classification']

        # Map the classifications to colors using the color_map
        colors = classifications.map(color_map)

         
        for idx, (x_attr, y_attr) in enumerate(zip(x_attributes, y_attributes)):
            ax = axes[idx]  # Access the correct subplot
            ax.scatter(merged_df[x_attr], merged_df[y_attr], c = colors)
            ax.set_xlabel(x_attr)
            ax.set_ylabel(y_attr)
            ax.set_title(f'{x_attr} vs {y_attr}')

        # Adjust layout for subplots
        plt.tight_layout()

        # Show the plots
        plt.show()


    def generate_statistics_image(self, row, background_color=(0, 0, 0), text_color=(255, 255, 255),
                                  text_size=0.5, text_thickness=1, text_position=(50, 50)):

       # Check if a statistics window is open and close it if it exists
        if self.statistics_window is not None:
            cv2.destroyWindow(self.statistics_window)
            self.statistics_window = None

        data = {'Attribute': [], 'Value': []}

        # Add the statistics to the DataFrame
        data['Attribute'].append('')
        data['Value'].append(os.path.dirname(row['File_Name']))
        data['Attribute'].append('')
        data['Value'].append('')
        for column, value in row.iloc[:-1].items():
            # Modify column names using column_mapping
            if column == 'File_Name':
                data['Attribute'].append(self.column_mapping['File_Name'] + ': ')
                data['Value'].append(os.path.basename(value))
            elif column in self.column_mapping:
                data['Attribute'].append(self.column_mapping[column] + ':')
                data['Value'].append(str(value))

        # Create the DataFrame from the data
        df = pd.DataFrame(data)

        # Define fixed widths for columns
        attribute_width = 30
        value_width = 15

        # Convert the DataFrame to a formatted table with fixed widths
        table_text = ''
        for attribute, value in zip(data['Attribute'], data['Value']):
            attribute = attribute[:attribute_width].rjust(attribute_width)
            value = value[:value_width].ljust(value_width)
            table_text += attribute + value + '\n'

        # Remove the trailing newline
        table_text = table_text.rstrip()        
        
        # Split the table text into lines
        text_lines = table_text.split('\n')

        # Calculate the dimensions of the statistics image
        attribute_offset = 20  # Offset for the attribute column
        value_offset = 200  # Offset for the value column

        # Calculate the maximum text width
        max_text_width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0][0]
            for line in text_lines
        )


        # Calculate the overlay coordinates on the resized image
        overlay_start = (text_position[0], text_position[1])
        overlay_end = (overlay_start[0] + value_offset + max_text_width - 5,
                       overlay_start[1] + len(text_lines) * attribute_offset + 30)
 
        # Create the statistics image
        statistics_image = np.zeros(
            (overlay_end[1] - overlay_start[1], overlay_end[0] - overlay_start[0], 3), dtype=np.uint8)
        statistics_image[:] = background_color

        # Draw the text onto the statistics image
        for i, line in enumerate(text_lines):
            if i == 0:
                # Handle the title line separately
                title_position = (0, (i + 1) * attribute_offset)
                cv2.putText(statistics_image, line, title_position, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, text_color, text_thickness)
            else:
                parts = line.split(":", maxsplit=1)
                if len(parts) == 2:
                    attribute, value = parts
                    # print(attribute, '-', value)
                else:
                    attribute = line.strip()
                    value = ""  # Set a default value if needed

                # Strip the first 10 characters from the attribute and remove underscores
                attribute = attribute[10:].strip('_')
                # print(attribute)
                
                attribute_position = (0, (i + 1) * attribute_offset)
                value_position = (value_offset, (i + 1) * attribute_offset)
                # print(attribute_position,value_position)

                cv2.putText(statistics_image, value, value_position, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, text_color, text_thickness)
                cv2.putText(statistics_image, attribute, attribute_position, cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, text_color, text_thickness)
        return statistics_image


    def get_sharpened_image(self, record_index):
        # Assuming you have a DataFrame 'images_pd' with a 'Sharpened_Image' column
        binary_data = self.sorted_pd.at[record_index, 'Sharpened_Image']

        # Convert the binary data to a NumPy array
        image_array = np.frombuffer(binary_data, np.uint8)

        # Decode the image using OpenCV
        sharpened_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if self.images_pd.at[record_index, 'Orientation'] == 'Landscape':
            sharpened_image, p = resize_image(sharpened_image, scale_percent=50)  # Adjust the scale percent as desired
        elif self.images_pd.at[record_index, 'Orientation'] == 'Portrait':
            sharpened_image, p = resize_image(sharpened_image, scale_percent=50)  # Adjust the scale percent as desired

        return sharpened_image


    def get_dataframe_records(self):
        return self.images_pd.values.tolist()    


    def create_dataframe(self, display_images, file_objects):
        # Your image processing code here
        scale = 20
        enhanced_actions = False
        batch_size = 25
        data_list = []

    # Populate the DataFrame with the data
        for image in display_images:
            print(f'---------------------- {image.image_id}')
            print(f'---------------------- {image.fname.full_path}')
            data_list.append({
                'Image_ID': image.image_id,
                'File_Name': image.fname.full_path,
                'Orientation': image.orientation,
                # print(f'---------------------- {image.file_objects.full_path}')
                'Brightness': round(image.image_stats.brightness),
                'Contrast': round(image.image_stats.contrast),
                'Haze_Factor': round(image.image_stats.haze_factor, 2),
                'Hough_Info': image.image_stats.hough_info[0],
                'Hough_Circles': 0,
                'Harris_Corners': image.image_stats.harris_corners,
                'Contour_Info': image.image_stats.contour_info[0],
                'Laplacian': round(image.image_stats.laplacian),
                'SHV': round(image.image_stats.shv, 2),
                'Variance': image.image_stats.variance,
                'Exposure': image.camera_settings.exposure,
                'F_Stop': image.camera_settings.fstop,
                'ISO': image.camera_settings.iso,
                'Black_Pixels': image.image_stats.b_w[0],
                'Mid_tone_Pixels': image.image_stats.b_w[1],
                'White_Pixels': image.image_stats.b_w[2],
                'Faces': image.image_stats.faces,
                'Eyes': image.image_stats.eyes,
                'Bodies': image.image_stats.bodies,
                'Focal_Length': image.camera_settings.focal_length,
                'Classification': image.classification,
                'Original_Height': image.original_image_height,
                'Original_Width': image.original_image_width,
                'Scale_Down': image.scale_factor[0],
                'Scale_Up': image.scale_factor[1],

                'Original_Image': image.convertToBinaryData()[0],
                'Grayscale_Image': image.convertToBinaryData()[1],
                'Denoised_Image': image.convertToBinaryData()[2],
                'Dehazed_Image': image.convertToBinaryData()[3],
                'Sharpened_Image': image.convertToBinaryData()[4],





                # cv2.imshow("ddf", image.display_images.image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imshow("Gray Scale Image", image.image_gs)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imshow("Denoised Image", image.image_dnz)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imshow("Sharpened Image", image.image_sharp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()




                
                



            })


            # self.image_gs = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            # self.image_dnz = cv2.fastNlMeansDenoisingColored(self.image.astype(np.uint8),None,2,2,7,21)               

            # self.image_sharp = self.sharpen()






       # Create the DataFrame from the list of dictionaries
        self.images_pd = pd.DataFrame(data_list, columns = self.column_names)
        print('pass fname instead')
        # ImageProcessor.initialize(image.fname.full_path)

        return self.images_pd

    


    def audit_filing_system(self):
        pass


    def delete_files(self,files_to_delete):

        for file_to_delete in files_to_delete:
            try:
                os.remove(file_to_delete)
                # print(f"File {file_to_delete} deleted successfully.")
            except Exception as e:
                print(f"Error: {e}")
        print(f" {files_to_delete} Files associated with {ImageProcessor.parent_dir} deleted successfully.")

        return len(files_to_delete)



    def get_classification_for_original_image(self,file):
        # print(f'Parent Directory: {ImageProcessor.parent_dir}')
        # path = ImageProcessor.parent_dir + '/' + file
        # # print(f'Path: {path}')
        row = self.images_pd.loc[self.images_pd['File_Name'] == file]
        # print(f'Row: {row}')
        if not row.empty:
            classification = row['Classification'].iloc[0] 
            # classification = row['Classification'].value[0]
            # print(f'Classification check: {classification}')
            return classification
        else:
            return None  # Return None if the file name is not found in the DataFrame
    
    def filter_files(self, root):
        valid_files = []
        files_to_delete = []
        u_classifier = 0
        a_classifier = 0
        b_classifier = 0
        d_classifier = 0
        g_classifier = 0

        # print(f'{ImageProcessor.parent_dir}')
        # print(f'{root}\n')

        for subdir, dirs, files in os.walk(root):
            for file in files:
                if root == ImageProcessor.parent_dir:
                        if file.endswith('.JPG') and '$' not in file and 'r_' not in file and subdir == root:
                            valid_files.append(os.path.join(subdir, file))
                            classifier = self.get_classification_for_original_image(file)
                            # print(f'classifier = {classifier}')
                            if classifier == 'U':
                                u_classifier += 1
                            elif classifier == 'A':
                                a_classifier +=1
                            elif classifier == 'B':
                                b_classifier +=1
                            elif classifier == 'D':
                                d_classifier +=1
                            elif classifier == 'G':
                                g_classifier +=1


    def find_file_mismatches(self):
        pass



        # self.print_filing_stats(self.stats, self.distances_df)

    def select_images_by_classification(self, target_classification):
        # Select rows with the specified classification and specific columns
        selected_rows = self.images_pd.loc[self.images_pd['classification'] == target_classification, ['image_id', 'file_name', 'classification']]
        return selected_rows


    def process_and_save_accepted_images(self):
        accepted_folder = ImageProcessor.accepted_dir
        final_folder = os.path.join(accepted_folder, 'final')

        # Create the final directory if it doesn't exist
        os.makedirs(final_folder, exist_ok=True)

        # Get a list of all files in the accepted folder
        accepted_files = [file for file in os.listdir(ImageProcessor.accepted_dir) if file.endswith('.JPG')]
        # print(f'Accepted Files {accepted_files}')
        counter = 0
        for file_name in accepted_files:
            # Construct the full path to the accepted image
            accepted_image_path = os.path.join(ImageProcessor.accepted_dir, file_name)
            # print(f'New File Name and Path: {accepted_image_path}')
            # Read the image
            image = cv2.imread(accepted_image_path)

            # Apply denoise and sharpen procedures
            color_image_processor = Color_Image(image, None, None)
            # denoised_image = color_image_processor.denoise_image(image)
            sharpened_image = color_image_processor.image_sharp

            # Save the processed image to the final folder
            final_image_path = os.path.join(final_folder, file_name)
            cv2.imwrite(final_image_path, sharpened_image)
            counter = counter + 1
            # print(f'Saving {counter} of {len(accepted_files)}')

        # print(f"Processing and saving accepted images completed.")

  
    def update_image_brightness(self, brightness_levels):
        # Get slider values for each HSV channel
        hsv_image = cv2.cvtColor(self.grid_image, cv2.COLOR_BGR2HSV)
        convert 
        for v_value in brightness_levels:




            h_val = self.hue_slider.get()
            s_val = self.saturation_slider.get()
            v_val = self.value_slider.get()
            # Get other slider values for different parameters (similar to above)

            # Convert the image to HSV color space

            # Adjust HSV channels using the slider values
            hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + h_val, 0, 179)  # Hue channel (0-179)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + s_val, 0, 255)  # Saturation channel (0-255)
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + v_val, 0, 255)  # Value channel (0-255)
        # Adjust other HSV channels using other slider values (similar to above)

        # Convert the image back to BGR color space
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        # Display the adjusted image
        cv2.imshow("Adjusted Image", adjusted_image)



    # def modify_HSV(self,hsv):

            

    #         # if we haven't been given a defined filter, use the filter values from the GUI
    #         # if not hsv_filter
    #             # hsv_filter = self.get_hsv_filter_from_controls()

    #         # add/subtract saturation and value
    #         h, s, v = cv2.split(hsv)
    #         s = shift_channel(s, 0)
    #         s = shift_channel(s, 0)
    #         v = shift_channel(v, 50)
    #         v = shift_channel(v, 0)
    #         hsv = cv2.merge([h, s, v])

    #         # Set minimum and maximum HSV values to display
    #         lower = np.array([0,0,0])
    #         upper = np.array([255,255,255])
    #         # Apply the thresholds
    #         mask = cv2.inRange(hsv, lower, upper)
    #         result = cv2.bitwise_and(hsv, hsv, mask=mask)

    #         # convert back to BGR for imshow() to display it properly
    #         processed_image = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    #         return processed_image        


