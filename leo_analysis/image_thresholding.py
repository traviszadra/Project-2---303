# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:09:45 2026

@author: Leoooo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_map import plot_google_map
import math
import time
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from skimage.color import rgb2gray
from skimage import io, filters, measure, morphology

def road_array():
    #%% Grid setup
    Nx = 25
    Ny = 20 
    #dt = 1/480  # days
    #dt = 0.0025
    dt = 1/240
    Ndays = 365
    D = 10  # diffusion coefficient (mi²/day)
    Lx = Ly = 1
    
    lat = [45.5, 46.0]
    lon = [-111.6, -110.8]
    
    x = np.linspace(min(lon), max(lon), Nx) # setting intial x and y arrays for grid
    y = np.linspace(min(lat), max(lat), Ny)
    X, Y = np.meshgrid(x, y, indexing = 'ij')   # indexing through 'row, columns' like we set up matrices
    
    
    
    plt.figure(1)
    
    
    extent = [x.min(), x.max(), y.min(), y.max()]
    center_lat = np.mean(lat)
    center_lon = np.mean(lon)
    center_lat = np.mean(lat)
    aspect_fix = 1/np.cos(np.deg2rad(center_lat))
    
    map_img, _ = plot_google_map(
        lat=center_lat,
        lon=center_lon,
        zoom=10,
        size=(1024, 1024),
        maptype="roadmap",
                
        api_key="AIzaSyDUz4oSBuVc8LvjAqa26WARGJR9jw4-Ghk",
        return_image=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Show map background
    img = ax.imshow(map_img, extent=extent)
    plt.axis("off")
    plt.savefig("map.png")
    
    map = io.imread("map2.png")         # this is the manually modified map2.png
    #%% break into RGB
    
    RedImage = map[:, :, 0]
    GreenImage = map[:, :, 1]
    BlueImage = map[:, :, 2]
    alpha = map[:, :, :3]
    
    RedImage2 = RedImage.astype(np.int16) - BlueImage.astype(np.int16)
    GreenImage2 = GreenImage.astype(np.int16) - BlueImage.astype(np.int16)
    BlueImage2 = RedImage.astype(np.int16) - GreenImage.astype(np.int16)
    GreyImage2 = rgb2gray(alpha)
    
    # set color threshold
    
    plt.figure(11)
    level1 = filters.try_all_threshold(RedImage2,figsize=(10, 8), verbose=False)       # leaving this in as an example of how I determined which filtering algo to use
    plt.title("Testing different filter threshold algorithms")
    plt.show()
    
    level2 = filters.threshold_isodata(GreenImage2)
    
    level3 = filters.threshold_yen(BlueImage2)
    
    level4 = filters.threshold_otsu(GreyImage2)
    
    # RedMask = RedImage2 > level1
    GreenMask = GreenImage2 > level2
    BlueMask = BlueImage2 > level3
    GreyMask = GreyImage2 > level4
    
    plt.figure(1)
    
    plt.subplot(1,3,1)
    plt.imshow(GreyMask, cmap = 'grey')
    plt.title("Grey Map - icons")
    
    plt.subplot(1,3,2)
    plt.imshow(GreenMask, cmap = "grey")
    plt.title("Three inital masks used (units: pixles)\nIsodata filtered green - roads")
    
    plt.subplot(1,3,3)
    plt.imshow(BlueMask, cmap = "grey")
    plt.title("Yen Filtered Blue - rivers")
    
    plt.tight_layout()
    
    plt.show()
    
    #%% morphology to try to isolate the roads better and to fill gaps
    strel = morphology.disk(6)
    opened = morphology.opening(GreyMask, footprint = strel)
    
    strel = morphology.disk(6)
    closed_blue = morphology.opening(BlueMask, footprint = strel)
    
    for i in range(len(closed_blue[:,0])):           # need to invert colors
        for j in range(len(closed_blue[0,:])):
            if closed_blue[i,j] == True:
                closed_blue[i,j] = False
            elif closed_blue[i,j] == False:
                closed_blue[i,j] = True
    
    

    
    plt.figure(2)
    plt.imshow(closed_blue)
    plt.title("closed blue - isolates rivers/lakes")
    plt.show()
    
    green_minus_morphblue =  opened.astype(np.int16) -  GreenMask.astype(np.int16) - BlueMask.astype(np.int16)
    
    
    for i in range(len(GreenMask[:,0])):
        for j in range(len(GreenMask[0,:])):
            if green_minus_morphblue[i,j] < 0 :
                green_minus_morphblue[i,j] = 0
    
    
    plt.figure(3)
    plt.imshow(green_minus_morphblue, cmap = 'grey')
    plt.title("green minus morphed blue - further isolates rivers")
    plt.show()
    
    green_minus_morph = (opened.astype(np.int16) -  GreenMask.astype(np.int16))  - green_minus_morphblue
    
    # ^ this starts to bring all the masks together - the opened grey mask (icons), the green mask (roads), and the blue mask (rivers)
    
    for i in range(len(GreenMask[:,0])):
        for j in range(len(GreenMask[0,:])):
            if green_minus_morph[i,j] < 0 :
                green_minus_morph[i,j] = 0
                
    # additional morphology is done to try to sharpen the image of the roads
    
    strel = morphology.disk(1)
    green_minus_morph = morphology.opening(green_minus_morph, footprint = strel) 
    
    
    roads = green_minus_morph[20:685,73:815] #trim image to remove legends
    
    # Needed to convert from pixels to the high res array that travis made
    
    low_res = np.zeros((400 , 500))
    
    x_scaling = len(roads[:,0])/len(low_res[:,0])
    y_scaling = len(roads[0,:])/len(low_res[0,:])
   
   
    for i in range(len(roads[:,0])):
        for j in range(len(roads[0,:])):
            if roads[i,j] ==1:
                low_res[int(i//x_scaling), int(j//y_scaling)] = 1
              
    plt.figure(10)
    plt.imshow(low_res)
    plt.title("roads")
    plt.axis("off")
    plt.show()
    
    roads = np.flipud(low_res).T
    return roads


    
road_array()

