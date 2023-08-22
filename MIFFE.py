#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:40:01 2023

@author: vincentchouinard

---------------------------------------------------
MIFFE stands for Most Incredible Function File Ever
---------------------------------------------------
"""

import numpy as np
from PIL import Image
import os
import sys
import tifffile as tf
from tqdm import tqdm
from PyQt5.QtWidgets import QFileDialog, QApplication
import imageio.v2 as imageio
from pathlib import Path


#Hardcoded paths needed for imported modules
CaImAn_path = r'/Users/vincentchouinard/Documents/GitHub/WideField/Chouine'
WFmovie_path = r'/Users/vincentchouinard/Documents/GitHub/Widefield-Imaging-analysis'

sys.path.insert(1, CaImAn_path)
from motion_correction_caiman import get_wf_mc_params, correct_motion_directory


sys.path.insert(1,WFmovie_path)
from WFmovie import WFmovie, create_channel, ioi_epsilon_pathlength





def search_path(path_type='folder'):

    from pathlib import Path
    
    if path_type == 'folder':
        app = QApplication([])
        folder_selected = QFileDialog.getExistingDirectory()
        print ("You chose: %s" % folder_selected)
        return folder_selected
    if path_type == 'file':
        app = QApplication([])
        file_selected = QFileDialog.getOpenFileName()[0]
        print("You chose: %s" % file_selected)
        return file_selected



def create_movies(data_path, channels, **kwargs):
    
    """ data_path: folderpath in which all files are (data, stim, json, metadata, etc).
    channels: list of colors wanted to be outputed. ('blue','green','red' or 'ir')
    kwargs: channels_exist, binning, normalize, gaussian_filter, gaussian_std, temp_med_filter, temp_med_width, 
    substract_background.
    All kwargs of the above are defaulted to False, channels_exist to False means it will create movies instead of 
    reaching for them. gaussian_std is hardcoded to 1, and temp_med_width to 3. Still changeable as kwargs.
    
    Function has no output, but it creates an object named 'movies'. Its content is a list of movies of the specified 
    colors listed in 'channels' argument. """

    data_path = Path(data_path)

    # Hardcoded values of every parameter
    channels_exist = False
    binning = False
    normalize = False
    gaussian_filter = False
    gaussian_std = 1
    temp_med_filter = False
    temp_med_width = 3
    substract_background = False
    background_path = None
    stim_path = False
    memmap = False
    
    # Update values from kwargs
    channels_exist = kwargs.get('channels_exist', channels_exist)
    binning = kwargs.get('binning', binning)
    normalize = kwargs.get('normalize', normalize)
    gaussian_filter = kwargs.get('gaussian_filter', gaussian_filter)
    gaussian_std = kwargs.get('gaussian_std', gaussian_std)
    temp_med_filter = kwargs.get('temp_med_filter', temp_med_filter)
    temp_med_width = kwargs.get('temp_med_width', temp_med_width)
    substract_background = kwargs.get('substract_background', substract_background)
    background_path = kwargs.get('background_path', background_path)
    stim_path = kwargs.get('stim_path', stim_path)
    memmap = kwargs.get('memmap', memmap)
    
    
    # Print values
    print(f"data_path: {data_path}")
    print(f"stim_path: {stim_path}")
    print(f"channels: {channels}")
    print(f"channels_exist: {channels_exist}")
    print(f"binning: {binning}")
    print(f"normalize: {normalize}")
    print(f"gaussian_filter: {gaussian_filter}")
    print(f"gaussian_std: {gaussian_std}")
    print(f"temp_med_filter: {temp_med_filter}")
    print(f"temp_med_width: {temp_med_width}")
    print(f"substract_background: {substract_background}")
    print(f"background_path: {background_path}")

    if substract_background is True:
        background_path = Path(background_path)
    # Create channels
    if channels_exist is False:
        for channel in channels:
            create_channel(folder_path=data_path,channel=channel)
    
    # No need for this, Jé changed it on main
    #if stim_path is True:
    #    split_data_path = data_path.parts
    #    folderName = split_data_path[-1]
    #    stimPath = data_path.joinpath(str(folderName) + '-stim_signal.npy')  
    
    # Create movies
    movies = []
    for channel in channels:
        movie = WFmovie(data_path,channel,memmap=memmap) #no more stim_file_path needed
        if substract_background is True:
            background = np.mean(np.load(background_path / 'data' / '0.npy'),axis=0)
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
    return movies



def generate_data_folder(movie, folderpath, channel, module_path = CaImAn_path, tif=False, CaImAn=False, numpy=False):

    #Creates a folder in which movie is transformed to
    Correction_folder_name = 'Correction'
    newfolderpath = "".join([folderpath,  '/', Correction_folder_name])
    newfolderpath = Path(newfolderpath)
    filename = "".join([channel, '_movie', '.tif'])

    
    if not os.path.exists(newfolderpath):
        os.makedirs(newfolderpath)
    
    if not tif and not CaImAn and not numpy:
        print('Correction folder created, but is empty. Dont forget to specify what has to be created (tif, CaIman, numpy)')
    
    if tif:
        filepath = Path(newfolderpath) / Path(filename)
        movie.convert_to_tif(filepath, convert16=True)
    
    if CaImAn:
        newfolderpath = Path(newfolderpath)
        parameters = get_wf_mc_params()
        CaImAn_folder_path = str(newfolderpath / '')
        correct_motion_directory(CaImAn_folder_path, parameters, keywords=['.tif'])
        
    if numpy:
        npyfilename = Path("".join(['corrected_', channel, '_movie', '.npy']))
        newnpypath = newfolderpath / npyfilename
        corrected_tif_filename = "".join(['corrected_', filename])
        corrected_tif_path = newfolderpath / corrected_tif_filename
        imageio_array = imageio.imread(corrected_tif_path)
        nparray = np.array(imageio_array, dtype=np.uint16)
        np.save(newnpypath, nparray)
        
    if tif and CaImAn and numpy:
        movie_npy = np.load(newnpypath)
        return movie_npy


def bin_pixels(frame, bin_size):
    height, width = frame.shape[:2]
    binned_height = height // bin_size
    binned_width = width // bin_size

    reshaped_frame = frame[:binned_height * bin_size, :binned_width * bin_size].reshape(binned_height, bin_size,
                                                                                        binned_width, bin_size)
    binned_frame = np.sum(reshaped_frame, axis=(1, 3), dtype=np.float32)
    binned_frame = binned_frame / (bin_size ** 2)

    return binned_frame

def convert_to_hb(path_green, path_red, output_path, baseline=None, bin_size=2):

    with tf.TiffFile(path_green) as tifG, tf.TiffFile(path_red) as tifR:
        R_green = tifG.pages
        R_red = tifR.pages

        num_frames = len(R_green)  # Assuming the frames are along the first dimensiona
        binned_frame = bin_pixels(R_green[0].asarray(), bin_size)
        frame_shape = binned_frame.shape
        stack_shape = (num_frames, frame_shape[0], frame_shape[1])

        lambda1 = 450  # nm
        lamba2 = 700  # nm
        npoints = 1000
        baseline_hbt = 100  # uM
        baseline_hbo = 60  # uM
        baseline_hbr = 40  # uM
        rescaling_factor = 1e6

        eps_pathlength = ioi_epsilon_pathlength(lambda1, lamba2, npoints, baseline_hbt, baseline_hbo, baseline_hbr,
                                                filter)
        Ainv = np.linalg.pinv(eps_pathlength) * rescaling_factor

        if baseline is None:
            data = tf.imread(path_green)
            norm_factor_G = np.mean(data, axis=0)
            norm_factor_G = bin_pixels(norm_factor_G, bin_size)
            data = tf.imread(path_red)
            norm_factor_R = np.mean(data, axis=0)
            norm_factor_R = bin_pixels(norm_factor_R, bin_size)
            data = None
        else:
            data = tf.imread(path_green)[baseline[0]:baseline[1]]
            norm_factor_G = np.mean(data, axis=0)
            norm_factor_G = bin_pixels(norm_factor_G, bin_size)
            data = tf.imread(path_red)[baseline[0]:baseline[1]]
            norm_factor_R = np.mean(data, axis=0)
            norm_factor_R = bin_pixels(norm_factor_R, bin_size)
            data = None

        with tf.TiffWriter(output_path + "/dHbO.tif", bigtiff=True) as writerHbO, tf.TiffWriter(
                output_path + "/dHbR.tif", bigtiff=True) as writerHbR, tf.TiffWriter(output_path + "/dHbT.tif",
                                                                                     bigtiff=True) as writerHbT:
            for i in tqdm(range(num_frames)):
                data_G = bin_pixels(R_green[i].asarray(), bin_size) / norm_factor_G
                data_R = bin_pixels(R_red[i].asarray(), bin_size) / norm_factor_R
                ln_green = -np.log(data_G.flatten())
                ln_red = -np.log(data_R.flatten())
                ln_R = np.concatenate((ln_green.reshape(1, len(ln_green)), ln_red.reshape(1, len(ln_green))))
                Hbs = np.matmul(Ainv, ln_R)
                d_HbO = Hbs[0].reshape(frame_shape)
                d_HbR = Hbs[1].reshape(frame_shape)
                np.nan_to_num(d_HbO, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(d_HbR, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                d_HbT = d_HbO + d_HbR

                writerHbO.write(np.float32(d_HbO), contiguous=True)
                writerHbR.write(np.float32(d_HbR), contiguous=True)
                writerHbT.write(np.float32(d_HbT), contiguous=True)

    return None

def normalize_by_baseline(data, freq, baseline_time):
    """Normalize each frame by the baseline mean. baseline_time is in seconds.
    Baseline frames are taken at the start of the acquisition."""
    
    nframes = int(freq*baseline_time)
    data = data / np.mean(data[0:nframes], axis=0)
    return data


import cv2
import tifffile as tiff
import numpy as np
from tqdm import tqdm


def convert_to_grayscale(input_file, output_file, frame_range):
    video = cv2.VideoCapture(input_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file_template = output_file.replace('.tif', '_{}.tif')

    current_frame = frame_range[0]
    file_counter = 0

    while current_frame < frame_range[1]:
        output_file = output_file_template.format(file_counter)
        print(output_file)
        with tiff.TiffWriter(output_file, bigtiff=True) as tiff_writer:
            for _ in tqdm(range(3000)):
                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = video.read()

                frame = frame[:, 420:1500]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tiff_writer.write(frame, contiguous=True)

                current_frame += 1
                if current_frame > frame_range[1]:
                    break

            file_counter += 1

    video.release()


def create_hb_data(g_path, r_path, data_path, baseline, directory = "Hb_data", bin_size = 2):
    # Directory
    parent_dir = data_path

    # Path
    save_path = os.path.join(parent_dir, directory)

    if not os.path.exists(save_path):
        # Create the directory
        os.mkdir(save_path)
        print("Directory '% s' created." % directory)

    else:
        print("Directory '% s' already exists." % directory)

    current_path = os.getcwd()
    os.chdir(WFmovie_path)
    convert_to_hb(g_path, r_path, save_path, baseline=baseline, bin_size=bin_size)
    os.chdir(current_path)

    print("Done! dHbO/dHbR/dHbT .tifs are now saved in Hb_data.")


def fit_signal(signal1, signal2):
    """Fit one-dimensional signal1 to signal2 using simple inversion of a linear matrix equation.
    Returns: fit coefficients (a, b) and fit signal a * signal1 + b."""
    signal1 = np.expand_dims(signal1.flatten(), axis=1)  # (N x 1) vector
    signal2 = np.expand_dims(signal2.flatten(), axis=1)  # (N x 1) vector
    vectors = np.concatenate([signal1, np.ones((signal1.shape[0], 1))], axis=1)  # (N x 2) matrix
    coeffs = (np.linalg.pinv(vectors) @ signal2).flatten()
    fit = coeffs[0] * signal1 + coeffs[1]
    return coeffs, fit