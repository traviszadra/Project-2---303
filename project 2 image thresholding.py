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
    
    map = io.imread("map2.png")
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
    
    # plt.figure(11)
    # level1 = filters.try_all_threshold(RedImage2,figsize=(10, 8), verbose=False)
    # plt.show()
    
    level2 = filters.threshold_isodata(GreenImage2)
    
    level3 = filters.threshold_yen(BlueImage2)
    
    # plt.figure(14)
    level4 = filters.threshold_otsu(GreyImage2)
    # plt.show()
    
    # RedMask = RedImage2 > level1
    GreenMask = GreenImage2 > level2
    BlueMask = BlueImage2 > level3
    GreyMask = GreyImage2 > level4
    
    # plt.figure(1)
    
    # plt.subplot(1,3,1)
    # plt.imshow(GreyMask, cmap = 'grey')
    # plt.title("Grey Map")
    
    # plt.subplot(1,3,2)
    # plt.imshow(GreenMask, cmap = "grey")
    # plt.title("Isodata filtered green mask")
    
    # plt.subplot(1,3,3)
    # plt.imshow(BlueMask, cmap = "grey")
    # plt.title("Yen Filtered Blue mask")
    
    # plt.show()
    
    
    strel = morphology.disk(6)
    opened = morphology.opening(GreyMask, footprint = strel)
    
    strel = morphology.disk(6)
    closed_blue = morphology.opening(BlueMask, footprint = strel)
    
    for i in range(len(closed_blue[:,0])):           # need to invert
        for j in range(len(closed_blue[0,:])):
            if closed_blue[i,j] == True:
                closed_blue[i,j] = False
            elif closed_blue[i,j] == False:
                closed_blue[i,j] = True
    
    
    # plt.figure(6)
    # plt.imshow(opened)
    # plt.title("opened grey")
    # plt.show()
    
    # plt.figure(10)
    # plt.imshow(closed_blue)
    # plt.title("closed blue")
    # plt.show()
    
    green_minus_morphblue =  opened.astype(np.int16) -  GreenMask.astype(np.int16) - BlueMask.astype(np.int16)
    
    
    for i in range(len(GreenMask[:,0])):
        for j in range(len(GreenMask[0,:])):
            if green_minus_morphblue[i,j] < 0 :
                green_minus_morphblue[i,j] = 0
    
    
    # plt.figure(9)
    # plt.imshow(green_minus_morphblue, cmap = 'grey')
    # plt.title("green - morphblue")
    # plt.show()
    
    green_minus_morph = (opened.astype(np.int16) -  GreenMask.astype(np.int16))  - green_minus_morphblue
    
    
    for i in range(len(GreenMask[:,0])):
        for j in range(len(GreenMask[0,:])):
            if green_minus_morph[i,j] < 0 :
                green_minus_morph[i,j] = 0
                
    strel = morphology.disk(1)
    green_minus_morph = morphology.opening(green_minus_morph, footprint = strel)
    
    # strel = morphology.disk(8)
    # green_minus_morph = morphology.closing(green_minus_morph, footprint = strel)
    
                
    # plt.figure(8)
    # plt.imshow(green_minus_morph)
    
    roads = green_minus_morph[20:685,73:815] #trim image to remove legends
    
    # plt.figure(9)
    # plt.imshow(roads)
    # plt.title("roads")
    # plt.show()
    
    low_res = np.zeros((500 , 400))
    
    x_scaling = len(roads[:,0])/len(low_res[:,0])
    y_scaling = len(roads[0,:])/len(low_res[0,:])
    
    for i in range(len(roads[:,0])):
        for j in range(len(roads[0,:])):
            if roads[i,j] ==1:
                low_res[int(i//x_scaling), int(j//y_scaling)] = 1
              
    plt.figure(10)
    plt.imshow(low_res)
    plt.title("low res")
    plt.show()

    return(low_res)

#%% plotit

# plt.figure(2)

# extent = [x.min(), x.max(), y.min(), y.max()]
# center_lat = np.mean(lat)
# center_lon = np.mean(lon)
# center_lat = np.mean(lat)
# aspect_fix = 1/np.cos(np.deg2rad(center_lat))

# map_img, _ = plot_google_map(
#     lat=center_lat,
#     lon=center_lon,
#     zoom=10,
#     size=(1024, 1024),
#     maptype="roadmap",
            
#     api_key="AIzaSyDUz4oSBuVc8LvjAqa26WARGJR9jw4-Ghk",
#     return_image=True)

# fig, ax = plt.subplots(figsize=(8, 6))
# # Show map background
# img = ax.imshow(map_img, extent=extent)

# # study_car = 228000
# # boz_car = 32445
# # car_ratio = boz_car / study_car

# def source(t):
#     emissions = (100 * np.sin( (2*np.pi*t) - (np.pi/2) ) )+ 100
#     s = emissions * roads
#     return s
    
# Ndays = 1
# steps = Ndays * 24 + 1 #hourly

# s = source(1/2)
# plt.imshow(s, cmap = "binary", alpha = 0.6)
# plt.title(f"time: {i} hours")
    
# # for i in range(0, steps):

# #     s = source(i/24)
# #     plt.imshow(s, cmap = "binary", alpha = 0.6)
# #     plt.title(f"time: {i} hours")
    
    
    
road_array()







