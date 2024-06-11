import pandas as pd
import os
from tkinter import messagebox
import sqlite3

class DataHandler:
    def __init__(self):
        self.data = None

    def check_xlsx_file(full_excel_file):
        return os.path.exists(full_excel_file)

    # def load_image_for_display(image_path, image_orientation):
    #     image = cv2.imread(image_path)

    #     # Resize the image using the resize_image function
    #     if image_orientation == 'Landscape':
    #         resized_image, p = resize_image(image, scale_percent=30)  # Adjust the scale percent as desired
    #     if image_orientation == 'Portrait':
    #         resized_image, p = resize_image(image, scale_percent=20)  # Adjust the scale percent as desired
    #     #print(resized_image.shape[:2])


    def confirm_save_or_overwrite(self, file_path):

        file_exists_msg = f"File {file_path}, already exists. Do you want to overwrite it?"
        file_not_exists_msg = f"File, {file_path}, does not exist. Do you want to save the data?"
        overwrite = False
        save_data = False


        if os.path.exists(file_path):
            # Prompt the user with a message box to confirm overwriting
            overwrite = messagebox.askyesno("File Exists", file_exists_msg)
            if not overwrite:
                print("Operation canceled. Existing file will not be overwritten.")
                return False
        else:
            # Prompt the user with a message box to confirm saving the data
            save_data = messagebox.askyesno("File Does Not Exist", file_not_exists_msg)
            if not save_data:
                print("Data not saved.")
                return False
        return True


    def load_from_excel(self, dir_path, excel_file_path, sheet):
        print('here')
        try:
            # Read data from Excel into a DataFrame
            images_pd = pd.read_excel(excel_file_path, sheet_name = sheet)
            print(f"Data loaded from Excel: {excel_file_path}")
            #print(images_pd.head())

            # Additional processing or attribute setting here if needed

            return images_pd  # Return the loaded DataFrame

        except Exception as ex:
            print(f"Error loading data from Excel: {str(ex)}")
            return None


        


    def load_from_sql(self, sql_path, table_sheet):
        
        print(f'  *******************************************************************************************************') 
        print(f'  ** SQL Path: {sql_path}\n  ** Table or Sheet Name: {table_sheet}')
        print(f'  *******************************************************************************************************') 

        """
        Load data from a SQL file (SQLite database) and store it in the class instance's data attribute.
        Args:
            sql_file_path (str): Path to the SQL file (SQLite database file).
            table_name (str): Name of the SQL table to retrieve data from.

        Returns:
            None
        """
        try:
            # Establish a connection to the SQLite database
            conn = sqlite3.connect(sql_path)

            # Construct the SQL query to select data from the specified table
            query = f"SELECT * FROM {table_sheet}"
                
            
            # Read data from SQL into a DataFrame
            images_pd = pd.read_sql_query(query, conn)

            # Close the database connection
            conn.close()

            print(images_pd)
            return images_pd  # Return the loaded DataFrame


        except Exception as e:
            print(f"Error loading data from SQL: {str(e)}")


    def save_data_to_excel(self, images_pd, excel_file_path,sheet, confirm_req):


        if confirm_req:
            if not self.confirm_save_or_overwrite(excel_file_path):
                    print("Save operation canceled.")
                    
                    # Exit the function if the user cancels the save operation                    
                    return
        try:
            # Code that always runs, whether there was an exception or not
            columns_to_include = images_pd.columns[:-4]
            data_subset = images_pd[columns_to_include]                
            data_subset.to_excel(excel_file_path, sheet_name=sheet, index=False)
            print("Data saved successfully.")
    
        except Exception as e:
            print(f"Error saving data to Excel: {str(e)}")



            # data_handler.save_data_to_sql(self.images_pd, sql_path, table_sheet, confirm_req)



    def save_data_to_sql(self, images_pd, sql_path, table_sheet,confirm_req =True):

        print(f'  *******************************************************************************************************') 
        print(f'  ** SQL Path: {sql_path}\n  ** Table or Sheet Name: {table_sheet}')
        print(f'  *******************************************************************************************************') 
        










        if self.confirm_save_or_overwrite(sql_path):
            try:
                # Save to sql
                # print("data_handler class save to sql method", images_pd)
                print(f'***************** SQL File Path: {sql_path}\n***************** Sheet Name: {table_sheet}')

                # columns_to_include = images_pd.columns[:-4]
                # data_subset = images_pd[columns_to_include]                
                
                print(sql_path)
                # Connect to an SQLite database (or create a new one if it doesn't exist)
                conn = sqlite3.connect(sql_path)
                print(f'Connected')
                # Save the DataFrame to a table in the database
                images_pd.to_sql(name=table_sheet, con=conn, if_exists='replace', index=False)

                # Close the database connection
                conn.close()
                print('Done')

            except Exception as e:
                print(f"Error saving data to SQL: {str(e)}")

    

    def save_record_to_sql(self, record_to_save, sql_file_path, table):
        

        # Extract values from the record_to_save DataFrame
        image_id = record_to_save['Image_ID'].iloc[0]  # Assuming 'Image_ID' is the column name
        classification = record_to_save['Classification'].iloc[0]  # Assuming 'Classification' is the column name
        print(f'Updating Image_ID: {image_id} to Classification: {classification}')
        sql_query = f"UPDATE {table} SET Classification = '{classification}' WHERE Image_ID = '{image_id}';"

        # Connect to the database
        conn = sqlite3.connect(sql_file_path)

        # Create a cursor object
        cursor = conn.cursor()

        # Execute the SQL query
        cursor.execute(sql_query)

        # Commit the changes
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

