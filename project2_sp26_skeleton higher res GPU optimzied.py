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
#from indiv_analysis import travisanalysis, haydenanalysis


start = time.perf_counter() # Starting the timer
plt.close('all')

#%% Load wind data
data = pd.read_csv("wind2-1.csv")
wind_mean = data[' Mean Wind SpeedMPH'].values
wind_gust = data[' Max Gust SpeedMPH'].values
wind_dir = data[' WindDirDegrees'].values

#%% Grid setup

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


#%% calculate dx and dy as distances from the lat long data

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

#%% Model the emission source

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

#%% Establish the map and place markers for Bozeman and Belgrade

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

#%% Solve the advection problem for each time step
# Vectorize your operations for efficiency! Otherwise it will take forever.

steps = int(Ndays/dt)    # total number of steps for this simulation
steps_per_day = int(1 / dt)  

# initialize the concentration arrays

plot_freq = steps // (Ndays*24) # plot freq

C = cp.zeros((Nx,Ny)) + cp.asarray(S)
Cnew = C.copy()
#Chistory = np.zeros((steps // plot_freq, Nx, Ny), dtype=np.float32) # C_history stores the concentration info at each time step

n_snapshots = steps // plot_freq
Chistory = np.lib.format.open_memmap(
    r'C:\Users\travi\OneDrive - Montana State University\College Classes\2026 Semester 6\EMEC 303  CAEIII - Systems Analysis\Projects\Project 2\Chistory rescale20\Chistory.npy',
    dtype=np.float32, mode='w+', shape=(n_snapshots, Nx, Ny))

t = 0 # initial value for time

frac = np.zeros(steps)
timearr = np.linspace(0,steps,steps)

#pre computed GPU arrays
n_arr        = np.arange(steps)
day_idx_arr  = n_arr // steps_per_day
frac_arr     = (n_arr % steps_per_day) / steps_per_day

next_idx_arr = np.clip(day_idx_arr + 1, 0, 364)

wind_spd_arr = (1 - frac_arr) * wind_mean[day_idx_arr] + frac_arr * wind_mean[next_idx_arr]
wind_dir_arr = (1 - frac_arr) * wind_dir[day_idx_arr]  + frac_arr * wind_dir[next_idx_arr]

u_arr = cp.asarray(-24 * wind_spd_arr * np.sin(np.radians(wind_dir_arr)))
v_arr = cp.asarray(-24 * wind_spd_arr * np.cos(np.radians(wind_dir_arr)))

for n in range(steps):
    start1 = time.perf_counter()
#     t = t + dt  # in days
  
#     day_idx = n // steps_per_day
#     frac    = (n % steps_per_day) / steps_per_day
       
#     if day_idx < 364:
#     #Wind interpolation
#         wind_spd_val = (1 - frac) * wind_mean[day_idx] + frac * wind_mean[day_idx +1]
#         wind_dir_val = (1 - frac) * wind_dir[day_idx] + frac * wind_dir[day_idx+1]
        
#     else:
#         wind_spd_val = wind_mean[day_idx]
#         wind_dir_val = wind_dir[day_idx]
        
#     u = -24 * wind_spd_val * np.sin(np.radians(wind_dir_val))  # mi/day
#     v = -24 * wind_spd_val * np.cos(np.radians(wind_dir_val))  
  
    u = u_arr[n]
    v = v_arr[n]
    
    # vectorized diffusion update, skipping the boundaries
    lap = D * ((C[:-2, 1:-1] - 2*C[1:-1, 1:-1] + C[2:, 1:-1])/dx**2 + (C[1:-1, :-2] - 2*C[1:-1, 1:-1] + C[1:-1, 2:])/dy**2)
  

        #%%
    # Upwind advection in x. Use vectorization techniques.
    
    xvel = cp.where(
    u < 0,
    u * (C[2:, 1:-1] - C[1:-1, 1:-1]) / dx,   # forward diff
    u * (C[1:-1, 1:-1] - C[:-2, 1:-1]) / dx    # backward diff
)
    # if u < 0:
    #     xvel = u * (C[2:, 1:-1] - C[1:-1, 1:-1]) / dx   # forward diff
        
    #         # if u < 0:
    #         #  xvelocity = (u / dx) * (C[i+1, j] - C[i, j])
            
    # else:
    #     xvel = u * (C[1:-1, 1:-1] - C[:-2, 1:-1]) / dx # backward diff

    # Upwind advection in y
    yvel = cp.where(
    v < 0,
    v * (C[1:-1, 2:] - C[1:-1, 1:-1]) / dy,     # forward diff
    v * (C[1:-1, 1:-1] - C[1:-1, :-2]) / dy      # backward diff
)
    
    # if v < 0:
    #     yvel = v * (C[1:-1, 2:] - C[1:-1, 1:-1]) / dy   # forward diff
        
    # else:
    #     yvel = v * (C[1:-1, 1:-1] - C[1:-1, :-2]) / dy  # backward diff
        
        
    # Update interior
    Cnew[1:-1, 1:-1] = C[1:-1, 1:-1] + dt*(-xvel - yvel + lap + Sinterior)

    # Apply zero-flux boundary conditions
    Cnew[0, :] = Cnew[1,:]
    Cnew[-1, :] = Cnew[-2,:]
    Cnew[:, 0] = Cnew[:,1]
    Cnew[:, -1] = Cnew[:,-2]

    # Store previous state (as in your original code) and advance
    if n % plot_freq == 0:
        Chistory[n // plot_freq] = cp.asnumpy(C) #stores every plot freq step
    
    C = Cnew.copy()
    
    progress = n/steps*100
    
    end1 = time.perf_counter()
    time_elapsed = end1-start1
    
    percentvel = (100/steps) / time_elapsed 
    
    if n % 100: 
        print(f' We are {progress:0.4f}% done!')
        
        print(f' Currently doing {percentvel:0.4f}%/sec')
        
        percentremain = 100 - progress
        timedone = percentremain/percentvel / 60
        
        print(f'Done in {timedone:0.4f} minutes')
        
        print(f'Were on step {n} of {steps}')
    
    
end = time.perf_counter() # Starting the timer
time_elapsed = end-start

print(f"the elapsed time is {time_elapsed:0.2f}")

    





