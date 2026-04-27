# -*- coding: utf-8 -*-

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
dt = 1/240 #1/2000 
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
C = np.zeros((Nx,Ny)) + S
Cnew = C.copy()
Chistory = np.zeros((steps, Nx, Ny)) # C_history stores the concentration info at each time step

t = 0 # initial value for time

#frac = np.zeros(steps)
#time = np.linspace(0,steps,steps)

#%%
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
    Cnew[1:-1, 1:-1] = C[1:-1, 1:-1] + dt*(-xvel - yvel + lap + Sinterior)

    # Apply zero-flux boundary conditions
    Cnew[0, :] = Cnew[1,:]
    Cnew[-1, :] = Cnew[-2,:]
    Cnew[:, 0] = Cnew[:,1]
    Cnew[:, -1] = Cnew[:,-2]

    # Store previous state (as in your original code) and advance
    Chistory[n] = C.copy()
    C = Cnew.copy()
    
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
plot_freq = 10
frame_indices = list(range(0, steps, plot_freq))

ani = animation.FuncAnimation(
    fig, update, frames=frame_indices,
    interval=50, blit=False, repeat=False
)

#ani.save('C:/Users/travi/OneDrive - Montana State University/College Classes/2026 Semester 6/EMEC 303  CAEIII - Systems Analysis/Projects/Project 2/base_animation.mp4', fps=30, dpi=100, bitrate = 2000)


plt.show()
    
end = time.perf_counter() # Starting the timer
time_elapsed = end-start

print(f"the elapsed time is {time_elapsed:0.2f}")






#%% Check Criteria 1 and 2 for Belgrade, Bozeman, Four Corners, and Bridger (Hayden Analysis)
# Yearly average with wind
valley_mean_wind = np.mean(Chistory)
if valley_mean_wind > 53:
    print('Assuming average wind, the yearly average NO2 emissions for the Gallatin Valley is not met.')
    print(f"Average yearly NO2 emissions = {valley_mean_wind:0.2f} ppb")
    
else:
    print('Assuming average wind, the yearly average NO2 emissions for the Gallatin Valley is met.')
    print(f"Average yearly NO2 emissions = {valley_mean_wind:0.2f} ppb")

# 98th percentile concentration yearly distribution

steps_per_hour = steps_per_day // 24
C_daily = Chistory[:steps].reshape(Ndays, steps_per_day, Nx, Ny) # all timesteps throughout the day
C_dt = C_daily.reshape(Ndays, 24, steps_per_hour, Nx, Ny) # time is broken in to days, hours, and timesteps in the hour
C_hourly = np.mean(C_dt, axis = 2) # mean concentration map for every hour
C_hourly_mean = np.mean(C_hourly, axis = (2,3)) # average concentration over entire map for every hour
C_daily_max = np.max(C_hourly_mean, axis=1) # max over 24 hours
C_p98 = np.percentile(C_daily_max, 98, axis=0) # 98th percentile

if C_p98 < 100:
    print('The 98th percentile of max concentration is met.')
    print(f"Concentration = {C_p98:0.2f} ppb")
    
else:
    print('The 98th percentile of max concentration is not met.')
    print(f"Concentration = {C_p98:0.2f} ppb")

## Belgrade
# Criteria 1
C_Bel = Chistory[:, 13, 10]
C_Bel_mean = np.mean(C_Bel)

print(f"Average yearly Belgrade NO2 emissions = {C_Bel_mean:0.2f} ppb")

# Criteria 2
C_Bel_hourly = C_hourly[:, :, 13, 10]
C_Bel_daily_max = np.max(C_Bel_hourly, axis = 1)
C_Bel_p98 = np.percentile(C_Bel_daily_max, 98, axis=0)

print(f"Belgrade p98 concentration = {C_Bel_p98:0.2f} ppb")


## Bozeman
# Criteria 1
C_Boz = Chistory[:, 17, 7]
C_Boz_mean = np.mean(C_Boz)

print(f"Average yearly Bozeman NO2 emissions = {C_Boz_mean:0.2f} ppb")

# Criteria 2
C_Boz_hourly = C_hourly[:, :, 17, 7]
C_Boz_daily_max = np.max(C_Boz_hourly, axis = 1)
C_Boz_p98 = np.percentile(C_Boz_daily_max, 98, axis=0)

print(f"Bozeman p98 concentration = {C_Boz_p98:0.2f} ppb")


## Four Corners
# Criteria 1
C_FC = Chistory[:, 13, 7]
C_FC_mean = np.mean(C_FC)

print(f"Average yearly Four Corners NO2 emissions = {C_FC_mean:0.2f} ppb")

# Criteria 2
C_FC_hourly = C_hourly[:, :, 13, 7]
C_FC_daily_max = np.max(C_FC_hourly, axis = 1)
C_FC_p98 = np.percentile(C_FC_daily_max, 98, axis=0)

print(f"Four Corners p98 concentration = {C_FC_p98:0.2f} ppb")


## Bridger
# Criteria 1
C_Brid = Chistory[:, 20, 12]
C_Brid_mean = np.mean(C_Brid)

print(f"Average yearly Bridger NO2 emissions = {C_Brid_mean:0.2f} ppb")

# Criteria 2
C_Brid_hourly = C_hourly[:, :, 20, 12]
C_Brid_daily_max = np.max(C_Brid_hourly, axis = 1)
C_Brid_p98 = np.percentile(C_Brid_daily_max, 98, axis=0)

print(f"Bridger p98 concentration = {C_Brid_p98:0.2f} ppb")


# Graph Criteria
cities = ['Bozeman', 'Belgrade', 'Four Corners', 'Bridger Bowl']
city_means = [C_Boz_mean, C_Bel_mean, C_FC_mean, C_Brid_mean]
city_p98 = [C_Boz_p98, C_Bel_p98, C_FC_p98, C_Brid_p98]

plt.figure(3)
plt.bar(cities, city_means)
plt.axhline(y = 53, color = 'red', label = 'Limit (53 ppb)')
plt.xlabel('Areas')
plt.ylabel('Average NO2 Amount [ppb]')
plt.title('Average NO2 Amounts in Populated/Popular Areas')
plt.legend()
plt.show()

plt.figure(4)
plt.bar(cities, city_p98)
plt.axhline(y = 100, color = 'red', label = 'Limit (100 ppb)')
plt.xlabel('Areas')
plt.ylabel('98th Percentile Concentrations [ppb]')
plt.title('1 Hour Daily Maximum NO2 Concentrations in Populated/Popular Areas')
plt.legend()
plt.show()
#%% Carson Analysis
# Check worst days for each city, then shut off for worst days

top_days = 10

# Belgrade worst days
Bel10 = []
Bel_sorted = np.argsort(-C_Bel_daily_max)
Bel_idx = Bel_sorted[:top_days]
print(f'The {top_days} worst days in Belgrade are:')
for day in Bel_idx:
    Bel10.append((day, C_Bel_daily_max[day]))
    print(f'Day {day}: {C_Bel_daily_max[day]:0.2f}')

# Bozeman worst days
Boz10 = []
Boz_sorted = np.argsort(-C_Boz_daily_max)
Boz_idx = Boz_sorted[:top_days]
print(f'The {top_days} worst days in Bozeman are:')
for day in Boz_idx:
    Boz10.append((day, C_Boz_daily_max[day])) 
    print(f'Day {day}: {C_Boz_daily_max[day]:0.2f}')

# Four Corners worst days
FC10 = []
FC_sorted = np.argsort(-C_FC_daily_max)
FC_idx = FC_sorted[:top_days]
print(f'The {top_days} worst days in Four Corners are:')
for day in FC_idx:
    FC10.append((day, C_FC_daily_max[day])) 
    print(f'Day {day}: {C_FC_daily_max[day]:0.2f}')

# Bridger worst days
C_daily_mod = Chistory.reshape(Ndays, steps_per_day, Nx, Ny).copy()
C_daily_mod[80:346, :, 20, 12] = 0
for d in range(Ndays):
    if d % 7 not in [5,6]:
        C_daily_mod[d, :, 20, 12] = 0
C_dt_mod = C_daily_mod.reshape(Ndays, 24, steps_per_hour, Nx, Ny)
C_hourly_mod = np.mean(C_dt_mod, axis=2)
C_Brid_hourly = C_hourly_mod[:, :, 20, 12]
C_Brid_daily_max_mod = np.max(C_hourly_mod[:, :, 20, 12], axis=1)
Brid10 = []
Brid_sorted = np.argsort(-C_Brid_daily_max_mod)
Brid_idx = Brid_sorted[:10]
print(f'The {top_days} worst days in Bridger are:')
for day in Brid_idx:
    Brid10.append((day, C_Brid_daily_max_mod[day])) 
    print(f'Day {day}: {C_Brid_daily_max_mod[day]:0.2f}')

# Overall worst days
overall_worst = Bel10 + Boz10 + FC10 + Brid10
# make into array to use argsort
overall_days = np.array([i[0] for i in overall_worst])
overall_values = np.array([i[1] for i in overall_worst])

overall10 = []
overallsort = np.argsort(-overall_values)
overallidx = overallsort[:top_days]
print(f'The overall {top_days} worst days are:')
for i in overallidx:
    overall10.append((overall_days[i], overall_values[i]))
    print(f'Day {overall_days[i]}: {overall_values[i]:0.2f} ')

# Check Criteria 1 and 2 for Gallatin Valley

# 98th percentile concentration yearly distribution

steps_per_hour = steps_per_day // 24
C_daily_new = Chistory[:steps].reshape(Ndays, steps_per_day, Nx, Ny) # all timesteps throughout the day
C_daily_new = C_daily_new.copy() 
shutdown = [40, 129, 269, 270, 316, 317, 349, 355, 356, 363]
C_daily_new[shutdown, :, :, :] = 0
C_daily_new[80:346, :, 20, 12] = 0
for d in range(Ndays):
    if d % 7 not in [5,6]:
        C_daily_new[d, :, 20, 12] = 0
C_dt_new = C_daily_new.reshape(Ndays, 24, steps_per_hour, Nx, Ny) # time is broken in to days, hours, and timesteps in the hour
C_hourly_new = np.mean(C_dt_new, axis = 2) # mean concentration map for every hour
C_hourly_mean_new = np.mean(C_hourly_new, axis = (2,3)) # average concentration over entire map for every hour
C_daily_max_new = np.max(C_hourly_mean_new, axis=1) # max over 24 hours
C_p98_new = np.percentile(C_daily_max_new, 98, axis=0) # 98th percentile
Chistory_new = C_daily_new.reshape(steps, Nx, Ny)

# Check Criteria 1 and 2 for Belgrade, Bozeman, Bridger, and Four Corners

## Belgrade
# Criteria 1
C_Bel_new = Chistory_new[:, 13, 10]
C_Bel_mean_new = np.mean(C_Bel_new)

print(f"Average yearly Belgrade NO2 emissions = {C_Bel_mean_new:0.2f} ppb")

# Criteria 2
C_Bel_hourly_new = C_hourly_new[:, :, 13, 10]
C_Bel_daily_max_new = np.max(C_Bel_hourly_new, axis = 1)
C_Bel_p98_new = np.percentile(C_Bel_daily_max_new, 98, axis=0)

print(f"Belgrade p98 concentration = {C_Bel_p98_new:0.2f} ppb")


## Bozeman
# Criteria 1
C_Boz_new = Chistory_new[:, 17, 7]
C_Boz_mean_new = np.mean(C_Boz_new)

print(f"Average yearly Bozeman NO2 emissions = {C_Boz_mean_new:0.2f} ppb")

# Criteria 2
C_Boz_hourly_new = C_hourly_new[:, :, 17, 7]
C_Boz_daily_max_new = np.max(C_Boz_hourly_new, axis = 1)
C_Boz_p98_new = np.percentile(C_Boz_daily_max_new, 98, axis=0)

print(f"Bozeman p98 concentration = {C_Boz_p98_new:0.2f} ppb")


## Four Corners
# Criteria 1
C_FC_new = Chistory_new[:, 13, 7]
C_FC_mean_new = np.mean(C_FC_new)

print(f"Average yearly Four Corners NO2 emissions = {C_FC_mean_new:0.2f} ppb")

# Criteria 2
C_FC_hourly_new = C_hourly_new[:, :, 13, 7]
C_FC_daily_max_new = np.max(C_FC_hourly_new, axis = 1)
C_FC_p98_new = np.percentile(C_FC_daily_max_new, 98, axis=0)

print(f"Four Corners p98 concentration = {C_FC_p98_new:0.2f} ppb")

## Bridger
# Criteria 1
C_Brid_new = Chistory_new[:, 20, 12]
C_Brid_mean_new = np.mean(C_Brid_new)

print(f"Average yearly Bridger NO2 emissions = {C_Brid_mean_new:0.2f} ppb")

# Criteria 2
C_Brid_hourly_new = C_hourly_new[:, :, 20, 12]
C_Brid_daily_max_new = np.max(C_Brid_hourly_new, axis = 1)
C_Brid_p98_new = np.percentile(C_Brid_daily_max_new, 98, axis=0)

print(f"Bridger p98 concentration = {C_Brid_p98_new:0.2f} ppb")


# Graph Criteria
cities = ['Bozeman', 'Belgrade', 'Four Corners', 'Bridger Bowl']
city_means_new = [C_Boz_mean_new, C_Bel_mean_new, C_FC_mean_new, C_Brid_mean_new]
city_p98_new = [C_Boz_p98_new, C_Bel_p98_new, C_FC_p98_new, C_Brid_p98_new]

plt.figure(5)
plt.bar(cities, city_means_new)
plt.axhline(y = 53, color = 'red', label = 'Limit (53 ppb)')
plt.xlabel('Areas')
plt.ylabel('Average NO2 Amount [ppb]')
plt.title('Average NO2 Amounts in Populated/Popular Areas After Shutdown')
plt.legend()
plt.show()

plt.figure(6)
plt.bar(cities, city_p98_new)
plt.axhline(y = 100, color = 'red', label = 'Limit (100 ppb)')
plt.xlabel('Areas')
plt.ylabel('98th Percentile Concentrations [ppb]')
plt.title('1 Hour Daily Maximum NO2 Concentrations in Populated/Popular Areas After Shutdown')
plt.legend()
plt.show()

#%% Check Criteria 1 and 2 for Gallatin Valley (Hayden Analysis)

# Yearly average with wind
valley_mean_wind = np.mean(Chistory)
if valley_mean_wind > 53:
    print('Assuming average wind, the yearly average NO2 emissions for the Gallatin Valley is not met.')
    print(f"Average yearly NO2 emissions = {valley_mean_wind:0.2f} ppb")
    
else:
    print('Assuming average wind, the yearly average NO2 emissions for the Gallatin Valley is met.')
    print(f"Average yearly NO2 emissions = {valley_mean_wind:0.2f} ppb")

# 98th percentile concentration yearly distribution

steps_per_hour = steps_per_day // 24
C_daily = Chistory[:steps].reshape(Ndays, steps_per_day, Nx, Ny) # all timesteps throughout the day
C_dt = C_daily.reshape(Ndays, 24, steps_per_hour, Nx, Ny) # time is broken in to days, hours, and timesteps in the hour
C_hourly = np.mean(C_dt, axis = 2) # mean concentration map for every hour
C_hourly_mean = np.mean(C_hourly, axis = (2,3)) # average concentration over entire map for every hour
C_daily_max = np.max(C_hourly_mean, axis=1) # max over 24 hours
C_p98 = np.percentile(C_daily_max, 98, axis=0) # 98th percentile

if C_p98 < 100:
    print('The 98th percentile of max concentration is met.')
    print(f"Concentration = {C_p98:0.2f} ppb")
    
else:
    print('The 98th percentile of max concentration is not met.')
    print(f"Concentration = {C_p98:0.2f} ppb")


#%% Check Criteria 1 and 2 for Belgrade, Bozeman, and Four Corners (Hayden Analysis)

## Belgrade
# Criteria 1
C_Bel = Chistory[:, 13, 10]
C_Bel_mean = np.mean(C_Bel)

print(f"Average yearly Belgrade NO2 emissions = {C_Bel_mean:0.2f} ppb")

# Criteria 2
C_Bel_hourly = C_hourly[:, :, 13, 10]
C_Bel_daily_max = np.max(C_Bel_hourly, axis = 1)
C_Bel_p98 = np.percentile(C_Bel_daily_max, 98, axis=0)

print(f"Belgrade p98 concentration = {C_Bel_p98:0.2f} ppb")


## Bozeman
# Criteria 1
C_Boz = Chistory[:, 17, 7]
C_Boz_mean = np.mean(C_Boz)

print(f"Average yearly Bozeman NO2 emissions = {C_Boz_mean:0.2f} ppb")

# Criteria 2
C_Boz_hourly = C_hourly[:, :, 17, 7]
C_Boz_daily_max = np.max(C_Boz_hourly, axis = 1)
C_Boz_p98 = np.percentile(C_Boz_daily_max, 98, axis=0)

print(f"Bozeman p98 concentration = {C_Boz_p98:0.2f} ppb")


## Four Corners
# Criteria 1
C_FC = Chistory[:, 13, 7]
C_FC_mean = np.mean(C_FC)

print(f"Average yearly Four Corners NO2 emissions = {C_Boz_mean:0.2f} ppb")

# Criteria 2
C_FC_hourly = C_hourly[:, :, 13, 7]
C_FC_daily_max = np.max(C_FC_hourly, axis = 1)
C_FC_p98 = np.percentile(C_FC_daily_max, 98, axis=0)

print(f"Four Corners p98 concentration = {C_FC_p98:0.2f} ppb")

## Bridger
# Criteria 1
C_Brid = Chistory[:, 20, 12]
C_Brid_mean = np.mean(C_Brid)

print(f"Average yearly Bridger NO2 emissions = {C_Brid_mean:0.2f} ppb")

# Criteria 2
C_Brid_hourly = C_hourly[:, :, 20, 12]
C_Brid_daily_max = np.max(C_Brid_hourly, axis = 1)
C_Brid_p98 = np.percentile(C_Brid_daily_max, 98, axis=0)

print(f"Bridger p98 concentration = {C_Brid_p98:0.2f} ppb")


# Graph Criteria
cities = ['Bozeman', 'Belgrade', 'Four Corners', 'Bridger Bowl']
city_means = [C_Boz_mean, C_Bel_mean, C_FC_mean, C_Brid_mean]
city_p98 = [C_Boz_p98, C_Bel_p98, C_FC_p98, C_Brid_p98]

plt.figure(3)
plt.bar(cities, city_means)
plt.axhline(y = 53, color = 'red', label = 'Limit (53 ppb)')
plt.xlabel('Areas')
plt.ylabel('Average NO2 Amount [ppb]')
plt.title('Average NO2 Amounts in Populated/Popular Areas')
plt.legend()
plt.show()

plt.figure(4)
plt.bar(cities, city_p98)
plt.axhline(y = 100, color = 'red', label = 'Limit (100 ppb)')
plt.xlabel('Areas')
plt.ylabel('98th Percentile Concentrations [ppb]')
plt.title('1 Hour Daily Maximum NO2 Concentrations in Populated/Popular Areas')
plt.legend()
plt.show()

#%% Carson Analysis
# Check worst days for each city, then shut off for worst days

top_days = 10

# Belgrade worst days
Bel10 = []
Bel_sorted = np.argsort(-C_Bel_daily_max)
Bel_idx = Bel_sorted[:top_days]
print(f'The {top_days} worst days in Belgrade are:')
for day in Bel_idx:
    Bel10.append((day, C_Bel_daily_max[day]))
    print(f'Day {day}: {C_Bel_daily_max[day]:0.2f}')

# Bozeman worst days
Boz10 = []
Boz_sorted = np.argsort(-C_Boz_daily_max)
Boz_idx = Boz_sorted[:top_days]
print(f'The {top_days} worst days in Bozeman are:')
for day in Boz_idx:
    Boz10.append((day, C_Boz_daily_max[day])) 
    print(f'Day {day}: {C_Boz_daily_max[day]:0.2f}')

# Four Corners worst days
FC10 = []
FC_sorted = np.argsort(-C_FC_daily_max)
FC_idx = FC_sorted[:top_days]
print(f'The {top_days} worst days in Four Corners are:')
for day in FC_idx:
    FC10.append((day, C_FC_daily_max[day])) 
    print(f'Day {day}: {C_FC_daily_max[day]:0.2f}')

# Bridger worst days
C_daily_mod = Chistory.reshape(Ndays, steps_per_day, Nx, Ny).copy()
C_daily_mod[80:346, :, 20, 12] = 0
for d in range(Ndays):
    if d % 7 not in [5,6]:
        C_daily_mod[d, :, 20, 12] = 0
C_dt_mod = C_daily_mod.reshape(Ndays, 24, steps_per_hour, Nx, Ny)
C_hourly_mod = np.mean(C_dt_mod, axis=2)
C_Brid_hourly = C_hourly_mod[:, :, 20, 12]
C_Brid_daily_max_mod = np.max(C_hourly_mod[:, :, 20, 12], axis=1)
Brid10 = []
Brid_sorted = np.argsort(-C_Brid_daily_max_mod)
Brid_idx = Brid_sorted[:10]
print(f'The {top_days} worst days in Bridger are:')
for day in Brid_idx:
    Brid10.append((day, C_Brid_daily_max_mod[day])) 
    print(f'Day {day}: {C_Brid_daily_max_mod[day]:0.2f}')

# Overall worst days
overall_worst = Bel10 + Boz10 + FC10 + Brid10
# make into array to use argsort
overall_days = np.array([i[0] for i in overall_worst])
overall_values = np.array([i[1] for i in overall_worst])

overall10 = []
overallsort = np.argsort(-overall_values)
overallidx = overallsort[:top_days]
print(f'The overall {top_days} worst days are:')
for i in overallidx:
    overall10.append((overall_days[i], overall_values[i]))
    print(f'Day {overall_days[i]}: {overall_values[i]:0.2f} ')

# Check Criteria 1 and 2 for Gallatin Valley

# 98th percentile concentration yearly distribution

steps_per_hour = steps_per_day // 24
C_daily_new = Chistory[:steps].reshape(Ndays, steps_per_day, Nx, Ny) # all timesteps throughout the day
C_daily_new = C_daily_new.copy() 
shutdown = [40, 129, 269, 270, 316, 317, 349, 355, 356, 363]
C_daily_new[shutdown, :, :, :] = 0
C_daily_new[80:346, :, 20, 12] = 0
for d in range(Ndays):
    if d % 7 not in [5,6]:
        C_daily_new[d, :, 20, 12] = 0
C_dt_new = C_daily_new.reshape(Ndays, 24, steps_per_hour, Nx, Ny) # time is broken in to days, hours, and timesteps in the hour
C_hourly_new = np.mean(C_dt_new, axis = 2) # mean concentration map for every hour
C_hourly_mean_new = np.mean(C_hourly_new, axis = (2,3)) # average concentration over entire map for every hour
C_daily_max_new = np.max(C_hourly_mean_new, axis=1) # max over 24 hours
C_p98_new = np.percentile(C_daily_max_new, 98, axis=0) # 98th percentile
Chistory_new = C_daily_new.reshape(steps, Nx, Ny)

# Check Criteria 1 and 2 for Belgrade, Bozeman, Bridger, and Four Corners

## Belgrade
# Criteria 1
C_Bel_new = Chistory_new[:, 13, 10]
C_Bel_mean_new = np.mean(C_Bel_new)

print(f"Average yearly Belgrade NO2 emissions = {C_Bel_mean_new:0.2f} ppb")

# Criteria 2
C_Bel_hourly_new = C_hourly_new[:, :, 13, 10]
C_Bel_daily_max_new = np.max(C_Bel_hourly_new, axis = 1)
C_Bel_p98_new = np.percentile(C_Bel_daily_max_new, 98, axis=0)

print(f"Belgrade p98 concentration = {C_Bel_p98_new:0.2f} ppb")


## Bozeman
# Criteria 1
C_Boz_new = Chistory_new[:, 17, 7]
C_Boz_mean_new = np.mean(C_Boz_new)

print(f"Average yearly Bozeman NO2 emissions = {C_Boz_mean_new:0.2f} ppb")

# Criteria 2
C_Boz_hourly_new = C_hourly_new[:, :, 17, 7]
C_Boz_daily_max_new = np.max(C_Boz_hourly_new, axis = 1)
C_Boz_p98_new = np.percentile(C_Boz_daily_max_new, 98, axis=0)

print(f"Bozeman p98 concentration = {C_Boz_p98_new:0.2f} ppb")


## Four Corners
# Criteria 1
C_FC_new = Chistory_new[:, 13, 7]
C_FC_mean_new = np.mean(C_FC_new)

print(f"Average yearly Four Corners NO2 emissions = {C_FC_mean_new:0.2f} ppb")

# Criteria 2
C_FC_hourly_new = C_hourly_new[:, :, 13, 7]
C_FC_daily_max_new = np.max(C_FC_hourly_new, axis = 1)
C_FC_p98_new = np.percentile(C_FC_daily_max_new, 98, axis=0)

print(f"Four Corners p98 concentration = {C_FC_p98_new:0.2f} ppb")

## Bridger
# Criteria 1
C_Brid_new = Chistory_new[:, 20, 12]
C_Brid_mean_new = np.mean(C_Brid_new)

print(f"Average yearly Bridger NO2 emissions = {C_Brid_mean_new:0.2f} ppb")

# Criteria 2
C_Brid_hourly_new = C_hourly_new[:, :, 20, 12]
C_Brid_daily_max_new = np.max(C_Brid_hourly_new, axis = 1)
C_Brid_p98_new = np.percentile(C_Brid_daily_max_new, 98, axis=0)

print(f"Bridger p98 concentration = {C_Brid_p98_new:0.2f} ppb")


# Graph Criteria
cities = ['Bozeman', 'Belgrade', 'Four Corners', 'Bridger Bowl']
city_means_new = [C_Boz_mean_new, C_Bel_mean_new, C_FC_mean_new, C_Brid_mean_new]
city_p98_new = [C_Boz_p98_new, C_Bel_p98_new, C_FC_p98_new, C_Brid_p98_new]

plt.figure(5)
plt.bar(cities, city_means_new)
plt.axhline(y = 53, color = 'red', label = 'Limit (53 ppb)')
plt.xlabel('Areas')
plt.ylabel('Average NO2 Amount [ppb]')
plt.title('Average NO2 Amounts in Populated/Popular Areas')
plt.legend()
plt.show()

plt.figure(6)
plt.bar(cities, city_p98_new)
plt.axhline(y = 100, color = 'red', label = 'Limit (100 ppb)')
plt.xlabel('Areas')
plt.ylabel('98th Percentile Concentrations [ppb]')
plt.title('1 Hour Daily Maximum NO2 Concentrations in Populated/Popular Areas')
plt.legend()
plt.show()