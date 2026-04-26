# -*- coding: utf-8 -*-

#%% Imports
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from plot_map import plot_google_map
import math
import time
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
from indiv_analysis import travisanalysis, haydenanalysis


start = time.perf_counter() # Starting the timer
plt.close('all')

# Load wind data
data = pd.read_csv("wind2-1.csv")
wind_mean = data[' Mean Wind SpeedMPH'].values
wind_gust = data[' Max Gust SpeedMPH'].values
wind_dir = data[' WindDirDegrees'].values

# Grid setup

rescale = 20
rescale_time = 175

Nx = 25 * rescale
Ny = 20  * rescale
#dt = 1/480  # days
#dt = 0.0025
dt = (1/240) / rescale_time
Ndays = 365
D = 10  # diffusion coefficient (mi²/day)

lat = [45.5, 46.0]
lon = [-111.6, -110.8]


x = np.linspace(min(lon), max(lon), Nx) # setting intial x and y arrays for grid
y = np.linspace(min(lat), max(lat), Ny)
X, Y = np.meshgrid(x, y, indexing = 'ij')   # indexing through 'row, columns' like we set up matrices


# calculate dx and dy as distances from the lat long data

def latlon2dist(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(np.radians(dlat / 2))**2 + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(np.radians(dlon / 2))**2
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1 - a))
    R = 3961  # radius of Earth in miles
    return R * c

miles_x = latlon2dist(np.mean(lat), lon[0], np.mean(lat), lon[1])
miles_y = latlon2dist(lat[0], np.mean(lon), lat[1], np.mean(lon)) 

dx = miles_x / (Nx - 1) #x[1] - x[0]
dy = miles_y / (Ny - 1) #y[1] - y[0]

# Model the emission source

x0, y0 = -111.073837, 45.817315 # have students fill this in
A = 3.3e4                           # peak ppb
sigma = 0.01

def source(x, y):
    return A * np.exp(-(((x - x0)**2 + (y - y0)**2) / (2*sigma**2)))

# Compute the source field for the whole map

S = source(X[:,:],Y[:,:])
Sinterior = source(X[1:-1,1:-1],Y[1:-1,1:-1])

# converted to cupy array:
S = cp.asarray(S)
Sinterior = cp.asarray(Sinterior)

# Establish the map and place markers for Bozeman and Belgrade

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
    maptype="terrain",
    markers=[
        (y0, x0, "red"),           # original source
        (45.679, -111.042, "blue"),     # bozeman
        (45.776, -111.176, "green"),   # belgrade
        (45.817, -110.897, "magenta")],  # bridger
            
    api_key="AIzaSyDUz4oSBuVc8LvjAqa26WARGJR9jw4-Ghk",
    return_image=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Show map background
img = ax.imshow(map_img, extent=extent, aspect=aspect_fix)

# plot the source on top of this map
hm = ax.imshow(
    S.get().T,                      
    extent=extent,            
    aspect=aspect_fix,        # fix degree anisotropy
    origin='lower',           # match Google tile orientation
    alpha=0.5,                # transparency so map is visible
    cmap='viridis',           
    interpolation='bilinear'  
)

cbar = plt.colorbar(hm, ax=ax, label='Source S (ppb)')
cbar.formatter = FormatStrFormatter('%.0e')  # <-- key line
cbar.update_ticks()

# Title and labels
ax.set_title('Map of the region')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.show()

# Solve the advection problem for each time step
# Vectorize your operations for efficiency! Otherwise it will take forever.

steps = int(Ndays/dt)    # total number of steps for this simulation
steps_per_day = int(1 / dt)  

# initialize the concentration arrays

plot_freq = steps // (Ndays*5) # plot freq

C = cp.zeros((Nx,Ny)) + cp.asarray(S)
Cnew = C.copy()

Chistory = np.load(r'C:\Users\travi\OneDrive - Montana State University\College Classes\2026 Semester 6\EMEC 303  CAEIII - Systems Analysis\Projects\Project 2\Chistory rescale20\Chistory.npy', mmap_mode='r')
#Chistory = np.zeros((steps // plot_freq, Nx, Ny), dtype=np.float32) # C_history stores the concentration info at each time step

t = 0 # initial value for time


    #%% plot an animation of the solution
fig, ax = plt.subplots(figsize=(8, 6))

# Show static map background
img = ax.imshow(map_img, extent=extent, aspect='auto')

# Show the pollution overlay
poll_img = ax.imshow(
    Chistory[0].T, extent=extent, origin='lower',
    cmap='jet', vmin=0, vmax=200, alpha=0.6
)

# Title and labels
title = ax.set_title("Animation of pollution across Gallatin Valley")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Create a colorbar
colorbar = fig.colorbar(poll_img, ax=ax, label="Concentration (ppb)")

# Update function for animation
def update(frame):
    t_day = frame / 24
    poll_img.set_data(Chistory[frame].T)

    # Interpolate wind info just like in main loop for the title
    day_idx = min(int(np.floor(t_day)), len(wind_mean) - 2)
    frac = t_day - int(np.floor(t_day))
    w_spd = (1 - frac) * wind_mean[day_idx] + frac * wind_mean[day_idx + 1]
    w_dir = (1 - frac) * wind_dir[day_idx] + frac * wind_dir[day_idx + 1]

    title.set_text(f"Day = {t_day:.1f}, Wind = {w_spd:.2f} mph, {w_dir:.0f}°")

    return [poll_img]

# Only use every plot_freq-th frame
frame_indices = list(range(1825))
#frame_indices = list(range(0, len(Chistory), 10))


ani = animation.FuncAnimation(
    fig, update, frames=frame_indices,
    interval=50, blit=False, repeat=False
)

ani.save('C:/Users/travi/OneDrive - Montana State University/College Classes/2026 Semester 6/EMEC 303  CAEIII - Systems Analysis/Projects/Project 2/upscaled_sim_animation.mp4', fps=30, dpi=100, bitrate = 2000)
#ani.save('C:/Users/travi/OneDrive - Montana State University/College Classes/2026 Semester 6/EMEC 303  CAEIII - Systems Analysis/Projects/Project 2/upscaled_sim_animation_basefps.mp4', fps=30, dpi=100, bitrate = 2000)

plt.show()
    
end = time.perf_counter() # Starting the timer
time_elapsed = end-start

print(f"the elapsed time is {time_elapsed:0.2f}")

#%% run individual analysis
from indiv_analysis import travisanalysis, travisanalysis2, haydenanalysis

#travisanalysis(Chistory, dt, lat, lon, S, plot_freq, x, y, wind_mean, wind_dir)

#travisanalysis2(Chistory, dt, lat, lon, S, plot_freq, x, y, wind_mean, wind_dir)

#haydenanalysis(Chistory, steps, steps_per_day, Nx, Ny, Ndays,rescale)






