
#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plot_map import plot_google_map
import math
import time
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter

start = time.perf_counter() # Starting the timer
plt.close('all')

#%% Load wind data
data = pd.read_csv("wind2-1.csv")
wind_mean = data[' Mean Wind SpeedMPH'].values
wind_gust = data[' Max Gust SpeedMPH'].values
wind_dir = data[' WindDirDegrees'].values

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


x = np.linspace(min(lat), max(lat), Nx) # setting intial x and y arrays for grid
y = np.linspace(min(lon), max(lon), Ny)
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

dx = x[1] - x[0]
dy = y[1] - y[0]

#%% Model the emission source

x0, y0 = 45.817315, -111.073837  # have students fill this in
A = 3.3e4                           # peak ppb
sigma = 0.01

def source(x, y):
    return A * np.exp(-(((x - x0)**2 + (y - y0)**2) / (2*sigma**2)))

# Compute the source field for the whole map

S = source(X[1:-1,1:-1],Y[1:-1,1:-1])

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
        (y[7], x[17], "blue"),     # bozeman
        (y[10], x[13], "green"),   # belgrade
        (y[12], x[20], "magenta")],  # bridger
            
    api_key="AIzaSyDUz4oSBuVc8LvjAqa26WARGJR9jw4-Ghk",
    return_image=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Show map background
img = ax.imshow(map_img, extent=extent, aspect=aspect_fix)

# plot the source on top of this map
hm = ax.imshow(
    S.T,                      
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
C = np.zeros((Nx,Ny)) 
Cnew = C.copy()
Chistory = np.zeros((steps, Nx,Ny)) # C_history stores the concentration info at each time step

t = 0 # initial value for time

#frac = np.zeros(steps)
#time = np.linspace(0,steps,steps)

for n in range(steps):
    t = t + dt  # in days
    
    day_idx = n // steps_per_day
    frac    = (n % steps_per_day) / steps_per_day
       
    if day_idx < 364:
    #Wind interpolation
        wind_spd_val = (1 - frac) * wind_mean[day_idx] + frac * wind_mean[day_idx +1]
        wind_dir_val = (1 - frac) * wind_dir[day_idx] + frac * wind_dir[day_idx+1]
        
    else:
        wind_spd_val = wind_mean[day_idx]
        wind_dir_val = wind_dir[day_idx]
        
    u = -24 * wind_spd_val * np.sin(np.radians(wind_dir_val))  # mi/day
    v = -24 * wind_spd_val * np.cos(np.radians(wind_dir_val))
    
    
    # vectorized diffusion update, skipping the boundaries
    lap = D * ((C[:-2, 1:-1] - 2*C[1:-1, 1:-1] + C[2:, 1:-1])/dx**2 + (C[1:-1, :-2] - 2*C[1:-1, 1:-1] + C[1:-1, 2:])/dy**2)
  

        #%%
    # Upwind advection in x. Use vectorization techniques.
    
    if u < 0:
        xvel = u * (C[2:, 1:-1] - C[1:-1, 1:-1]) / dx   # forward diff
   
        # if u < 0:
        #  xvelocity = (u / dx) * (C[i+1, j] - C[i, j])
        
    else:
        xvel = u * (C[1:-1, 1:-1] - C[:-2, 1:-1]) / dx # backward diff

    # Upwind advection in y
    if v < 0:
        yvel = v * (C[1:-1, 2:] - C[1:-1, 1:-1]) / dy   # forward diff
    else:
        yvel = v * (C[1:-1, 1:-1] - C[1:-1, :-2]) / dy  # backward diff

    # Update interior
    Cnew[1:-1, 1:-1] = C[1:-1, 1:-1] + dt*(-xvel -yvel + lap + S)

    # Apply zero-flux boundary conditions
    Cnew[0, :] = Cnew[1,:]
    Cnew[-1, :] = Cnew[-2,:]
    Cnew[:, 0] = Cnew[:,1]
    Cnew[:, -1] = Cnew[:,-2]

    # Store previous state (as in your original code) and advance
    Chistory[n] = C.copy()
    C = C.copy()
    
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
    t_day = frame * dt
    poll_img.set_data(Chistory[frame].T)

    # Interpolate wind info just like in main loop for the title
    day_idx = min(int(np.floor(t_day)), len(wind_mean) - 2)
    frac = t_day - int(np.floor(t_day))
    w_spd = (1 - frac) * wind_mean[day_idx] + frac * wind_mean[day_idx + 1]
    w_dir = (1 - frac) * wind_dir[day_idx] + frac * wind_dir[day_idx + 1]

    title.set_text(f"Day = {t_day:.1f}, Wind = {w_spd:.2f} mph, {w_dir:.0f}°")

    return [poll_img]

# Only use every plot_freq-th frame
plot_freq = 100
frame_indices = list(range(0, steps, plot_freq))

ani = animation.FuncAnimation(
    fig, update, frames=frame_indices,
    interval=50, blit=False, repeat=False
)

plt.show()
    
end = time.perf_counter() # Starting the timer
time_elapsed = end-start

print(f"the elapsed time is {time_elapsed:0.2f}")














