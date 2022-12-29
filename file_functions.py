import shutil
from PIL import Image, ImageStat
import os
import stat
from image_functions import *
from segment_class import *
import sqlite3
from timeit import default_timer as timer
import wx
from io import StringIO 












def delete_subs(subdir_name):

		try:
			shutil.rmtree(subdir_name + '/' + 'Good')
			shutil.rmtree(subdir_name + '/' + 'Bad')
		except:
			pass

def create_subs(subdir_name):

	print("Creating Directories...")
	path = subdir_name + '/' + 'Good'
	path_2= subdir_name + '/' + 'Bad'
	os.chmod( subdir_name, stat.S_IWRITE )
	os.makedirs(path)
	os.makedirs(path_2)






def	get_files(rootdir):

	fnames = []
	counter = 0
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
				split_tup = os.path.splitext(file)
				if split_tup[1] =='.JPG' and '$' not in(split_tup[0]) and 'r_' not in(split_tup[0]) and  rootdir == subdir:
			# '[Originals]' not in(subdir) and 'New Folder' \
			# 	not in(subdir) and 'Good' not in(subdir) and 'Bad' not in(subdir) and 'Bright' not in(subdir) and \
			# 	'Haze'  not in(subdir) and 'Dark' not in(subdir) and 'Rating 1' not in(subdir):
			# 		#print(subdir + '/' + file +  '\t' + str(os.path.getsize(subdir + '/' + file)))
					
					fn = [subdir, rootdir[13:], file]
					print(fn)
					fnames.append(File_Name(counter,fn))

					counter = counter + 1
	return fnames




def check_valid(fnames):

	c = 0
	d = 0
	for counter,fname in enumerate(fnames):
		try:
			img = Image.open(fname[3]).convert('RGB')  # open the image file
			img.verify() # verify that it is, in fact an image
			c = c + 1
			print('JPG Image - ', c)

		except (IOError, SyntaxError) as e:  
			d = d + 1
			print('Bad file:', d, fname[3]) # print out the names of corrupt files
			fnames.remove(fname[3])

def create_tbl(tbl_name):
	sqliteConnection = sqlite3.connect('D:/photo_info.db')

	create_tbl = """CREATE TABLE [""" + tbl_name + """] (
				image_id         INTEGER,
				file_path        TEXT,
				file_name        TEXT,
				orientation      TEXT,
				image_original   BLOB,
				image_gs         BLOB,
				image_dns        BLOB,
				image_dhz        BLOB,
				image_sharp      BLOB,
				brightness       DECIMAL,
				contrast         DECIMAL,
				haze_factor      DECIMAL,
				hough_lines      INTEGER,
				hough_length     INTEGER,
				contours         INTEGER,
				laplacian        DECIMAL,
				variance         DECIMAL,
				shv              DECIMAL,
				puter_says       BOOLEAN,
				status           TEXT,
				harris           INTEGER,
				final_status     TEXT,
				file_size        TEXT,
				camera_model     TEXT,
				file_date        TEXT,
				fstop            TEXT,
				shutter_speed    TEXT,
				file_orientation TEXT,
				file_xresolution TEXT,
				file_yresolution TEXT,
				file_focal_len   TEXT,
				file_iso         TEXT,
				lens_model       TEXT,
				firmware         TEXT,
				w                TEXT,
				l                TEXT,
				sn               TEXT)"""

	cursor = sqliteConnection.cursor()
	print(create_tbl)
	cursor.execute(create_tbl)
	cursor.close()
	sqliteConnection.close()

def shift_channel( c, amount):
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c

def convert_HSV(img):

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
 		# if not hsv_filter
			# hsv_filter = self.get_hsv_filter_from_controls()

        # add/subtract saturation and value
		h, s, v = cv2.split(hsv)
		s = shift_channel(s, sAdd)
		s = shift_channel(s, sSub)
		v = shift_channel(v, vAdd)
		v = shift_channel(v, vSub)
		hsv = cv2.merge([h, s, v])

		# Set minimum and maximum HSV values to display
		lower = np.array([hMin, sMin, vMin])
		upper = np.array([hMax, sMax, vMax])
		# Apply the thresholds
		mask = cv2.inRange(hsv, lower, upper)
		result = cv2.bitwise_and(hsv, hsv, mask=mask)

		# convert back to BGR for imshow() to display it properly
		processed_image = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

		return processed_image



def shift_channel(c, amount):
    print(c)
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c

def create_thumbnails(hsv_settings,image):

	thumbnails = []
	for i,setting in enumerate(hsv_settings):
		img_hsv = cv2.cvtColor(image.image_sharp.astype("uint8"), cv2.COLOR_BGR2HSV).astype("float32")
		(h,s,v) = cv2.split(img_hsv)
		s = s * setting[0]
		s = np.clip(s,0,255)
		v = v * setting[1]
		v = np.clip(v,0,255)
		img_hsv_1 = cv2.merge((h,s,v))
		img_hsv_2 = cv2.cvtColor(img_hsv_1.astype("uint8"), cv2.COLOR_HSV2BGR)
		if image.image.shape[0] > image.image.shape[1]:
			(img_hsv_2,orientation) = resize_image(img_hsv_2, 20)
		else:
			(img_hsv_2,orientation) = resize_image(img_hsv_2, 25)

		thumbnails.append(Thumbnail(h,s,v, img_hsv_2, str(i), setting[0], setting[1]))
		
	return thumbnails



def load_images(fnames,rootdir,scale,save):
	
	counter  = 0
	images = []
	for counter,fname in enumerate(fnames):
		if not counter:
			print(fname.f_parent_dir)
			#create_tbl(fname.f_parent_dir)
		(r_src,orientation) = resize_file(fname.full_path,scale)

		images.append(Color_Image(counter, fname.full_path,r_src,orientation))
		print('Loaded ' , counter + 1, 'of ', len(fnames), 'Images' )

	mean = np.mean([image.brightness for image in images])
	b_min = np.min([image.brightness for image in images])
	b_max = np.max([image.brightness for image in images])
	sd = np.std([image.brightness for image in images])	
	t_name = fnames[0].f_parent_dir	

	# named_window = []
	# named_windows = create_nw()

	d_s = [1.000,1.100,1.120,1.120, 1.120, 1.125, 1.130,1.135]	
	d_v = [1.000,1.250,1.280,1.310, 1.340, 1.340, 1.340,1.340]	
	hsv_settings = list(zip(d_s, d_v))


	"""  GUI Section with Image Thumbnails                     """

	app = wx.App()
	for counter,image in enumerate(images):
		# start = timer()
		thumbnails = create_thumbnails(hsv_settings,image)

		window = Window("BobL Images Image Processing",thumbnails,image.image_sharp)
		app_2 = wx.App()
		app.MainLoop()		


	"""  			END OF                    """
	"""  GUI Section with Image Thumbnails    """



	Color_Image.assign_grade(image)
	# Color_Image.put_text(image)
	if image.orientation == 'Portrait':
		image.image_sharp = cv2.rotate(image.image_sharp, cv2.ROTATE_90_COUNTERCLOCKWISE)


	Color_Image.save_out(image,rootdir,save, counter)
	# Color_Image.insertVaribleIntoTable(image,t_name,rootdir)



def create_segments(images):
	rows = 3
	columns = 1
	segments = []
	print('Creating Segments')
	for a, image in enumerate(images):
		grids = create_grids(image.image.shape[1],image.image.shape[0],columns,rows)
		for t,grid in (enumerate(grids)):
			d = len(grids)
			jj = np.copy(image.image_sharp[grid[0]: grid[0] + 373,grid[1]:grid[1] + 1680])
			# print(jj.shape)
			segments.append(Segment(image.image_id,image.fname,jj, t))
			print('Storing ' , t + 1, 'of ', d, 'for Image', a +1 )

	return segments



