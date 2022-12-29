# -*- coding: utf-8 -*
import shutil

import numpy as np

import cv2

from PIL import Image

from image_functions import *
from file_functions import *
from segment_class import *

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Rectangle

class CanvasWidget(Widget):
    def __init__(self, **kwargs):
        super(CanvasWidget, self).__init__(**kwargs)
        images = []
        #fnames = ['D:/Meaghan.jpg']
        #fnames = ['F:/Year 2022/2022-06-11 Briar Woods vs First Colonial Girls Soccer/353A5606.jpg']
        fnames = ['F:/Year 2022/Test/5Q2A5767.JPG']
 
        #fnames = ['F:/Year 2022/2022-11-11 Powhatan vs Dinwiddie Varsity Football/5Q2A5923.jpg']
        for counter,fname in enumerate(fnames):
            (r_src,orientation) = resize_file(fnames[counter],40)

            images.append(Color_Image(counter, fnames[counter],r_src,orientation))
            print('Loaded ' , counter + 1, 'of ', len(fnames), 'Images' )

        mean = np.mean([image.brightness for image in images])
        b_min = np.min([image.brightness for image in images])
        b_max = np.max([image.brightness for image in images])
        sd = np.std([image.brightness for image in images]) 
        print('bbb', images[0].orientation)

        # d_s = [1.000,1.100,1.120,1.120, 1.120, 1.125, 1.130,1.135]  
        # d_v = [1.000,1.250,1.280,1.310, 1.340, 1.340, 1.340,1.340]  

        d_s = [1.000,1.100,1.120,1.120, 1.120, 1.125]  
        d_v = [1.000,1.250,1.280,1.310,1.340, 1.340]  
        hsv_settings = list(zip(d_s, d_v))
        thumbnails = create_thumbnails(hsv_settings,images[0])
        imgs = []
        textures = []


        #Creates textures for all thumbnails
        for j,thumbnail in enumerate(thumbnails):
            imgs.append(cv2.cvtColor(thumbnails[j].image.astype('uint8'), cv2.COLOR_BGR2RGB))
            textures.append(Texture.create(size=(thumbnails[0].image.shape[1],
             thumbnails[0].image.shape[0]), colorfmt='bgr', bufferfmt='ubyte'))

        # for j, thumbnail in enumerate(thumbnails):
        #    cv2.imshow(str(j),thumbnail.image.astype('uint8'))

        for j in range(6):
            textures[j].blit_buffer(imgs[j].tostring(),colorfmt='rgb', bufferfmt='ubyte')
            textures[j].flip_vertical() 

            with self.canvas:
                #self.Color(1,1,1,1)
                if thumbnails[0].image.shape[0] > thumbnails[0].image.shape[1]:
                    self.rect = Rectangle(texture=textures[0], pos = (0,200),
                     size=(thumbnails[0].image.shape[1], thumbnails[0].image.shape[0]))
                    self.rect1 = Rectangle(texture=textures[1], pos = (400,200),
                     size=(thumbnails[1].image.shape[1], thumbnails[1].image.shape[0]))
                    self.rect2 = Rectangle(texture=textures[2], pos = (800,200),
                     size=(thumbnails[2].image.shape[1], thumbnails[2].image.shape[0]))
                    self.rect3 = Rectangle(texture=textures[3], pos = (1200,200),
                     size=(thumbnails[3].image.shape[1], thumbnails[3].image.shape[0]))
                    self.rect4 = Rectangle(texture=textures[4], pos = (1600,200),
                     size=(thumbnails[4].image.shape[1], thumbnails[4].image.shape[0]))
                    self.rect5 = Rectangle(texture=textures[5], pos = (2000,200),
                     size=(thumbnails[5].image.shape[1], thumbnails[5].image.shape[0]))
                else:
                    self.rect = Rectangle(texture=textures[0], pos = (100, 800),
                     size=(thumbnails[0].image.shape[1], thumbnails[0].image.shape[0]))
                    self.rect1 = Rectangle(texture=textures[1], pos = (875, 800),
                     size=(thumbnails[1].image.shape[1], thumbnails[1].image.shape[0]))
                    self.rect2 = Rectangle(texture=textures[2], pos = (1650, 800),
                     size=(thumbnails[2].image.shape[1], thumbnails[2].image.shape[0]))
                    self.rect3 = Rectangle(texture=textures[3], pos = (100,200),
                     size=(thumbnails[3].image.shape[1], thumbnails[3].image.shape[0]))
                    self.rect4 = Rectangle(texture=textures[4], pos = (875,200),
                     size=(thumbnails[4].image.shape[1], thumbnails[4].image.shape[0]))
                    self.rect5 = Rectangle(texture=textures[5], pos = (1650,200),
                     size=(thumbnails[5].image.shape[1], thumbnails[5].image.shape[0]))
 
                # self.bind(pos = self.update_rect, size = self.update_rect)

    def update_rect(self,*args):
        print('here kkkk')
        self.rect.pos = self.pos
        self.rect.size = self.size
  

class CanvasApp(App):
    def build(self):
        return CanvasWidget()

CanvasApp().run()           

