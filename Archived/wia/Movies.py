# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:38:43 2023

@author: je_gu
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import sys
import imageio
module_path = r'/Users/vincentchouinard/Documents/GitHub/Widefield-Imaging-analysis'
sys.path.insert(1,module_path)
from motion_correction_caiman import *
from WFmovie import WFmovie
from WFmovie import create_channel
from PyQt5.QtWidgets import QFileDialog, QApplication
import tifffile


#%% function to searchpath(). Tried with tkinter at first, but not compatible with MacOS internal DUI

def search_path():
    app = QApplication([])
    folder_selected = QFileDialog.getExistingDirectory()
    print ("You chose: %s" % folder_selected)
    return folder_selected

print("Where is your data folder?")
folderpath= search_path()

#%% Setup

data_path = Path(folderpath)
strpath = str(data_path)
#for i in np.arange(0,len(strpath)):
#    if strpath[i] == os.sep:
#        last_backslash_ind = i
#file = strpath[last_backslash_ind+1:-1] #Semble être inutile, et il manque la dernière lettre du file
channels_exist = False
channels = ['green'] #blue,green,red,ir
binning = False
normalize = False
gaussian_filter = False
gaussian_std = 1
temp_med_filter = False
temp_med_width = 3
substract_background = False
if substract_background is True:
    background_path = r'E:\Data Jeremie\Calcium imaging tests\M26\background'
#%% Create channels
if channels_exist is False:
    for channel in channels:
        create_channel(data_path,channel,r'experiment-metadata.json',binning)
#%% Create movies
movies = []
for channel in channels:
    movie = WFmovie(data_path,channel)
    if substract_background is True:
        background = np.mean(np.load(background_path+r'\data\0.npy'),axis=0)
        if binning is True:
            image = Image.fromarray(background)
            image = image.convert('F') #F = 32-bit floating point
            background = np.array(image.resize((movie.ncols,movie.nrows), resample=Image.Resampling.BILINEAR))
        movie.data = movie.data-background
    if normalize is True:
        movie.normalize_by_mean()
    if gaussian_filter is True:
        movie.gaussian_filt(gaussian_std)
    if temp_med_filter is True:
        movie.med_temp_filt(temp_med_width)
    movies.append(movie)
#%% Show movies
movie = movies[0]
#mask = movie.create_mask()
#movie.apply_mask(mask)
#moviearray = movie.convert_to_2d_matrix()

fig, ax = plt.subplots()
vmin = 0
vmax = 1000
framerate=1/movie.freq #le arg "interval" est en ms
def animate(i):
    ax.clear()
    plt.axis('off')
    ax.imshow(movie.data[i,:,:],vmin=vmin,vmax=vmax)
im = ax.imshow(movie.data[0],vmin=vmin,vmax=vmax)
cb = plt.colorbar(im)
anim = animation.FuncAnimation(fig, animate,frames=movie.nframes,interval=framerate)
plt.show()
#%% Saving movie in file


path = os.path.normpath(folderpath)
pathlist = path.split(os.sep)
filename = "".join([pathlist[-1], "_", channel, '_movie', '.tif'])
filepath = data_path / filename
movie.convert_to_tif(filepath)

#%% CaImAn

params = get_wf_mc_params()
folderpath = folderpath + '/'

#%%
correct_motion_directory(folderpath, params, keywords=['.tif'])

