# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:58:01 2026

@author: aweso
"""

#%% individual analysis Travis - comparing average houshold gas stove concentraction to concentraction see by plant.

def travisanalysis(Chistory, dt, lat, lon, S, plot_freq, x, y, wind_mean, wind_dir):
    import numpy as np
    import cupy as cp
    import matplotlib.pyplot as plt
    import pandas as pd
    from plot_map import plot_google_map
    import math
    import time
    import matplotlib.animation as animation
    from matplotlib.ticker import FormatStrFormatter
    
    x0, y0 = -111.073837, 45.817315 # have students fill this in
    A = 3.3e4                           # peak ppb
    sigma = 0.01
    
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
    
    start_TA = time.perf_counter() # Starting the timer
    
    C_avghousehold = 6 #ppb
    C_maxhousehold = 128 #ppb
    
    #C_avgpertime = Chistory.copy()
    C_avgpertime = (Chistory - C_maxhousehold)/(C_maxhousehold)
    
        # plot an animation of the solution
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Show static map background
    img = ax.imshow(map_img, extent=extent, aspect='auto')
    
    # Show the pollution overlay
    poll_img = ax.imshow(
        C_avgpertime[0].T, extent=extent, origin='lower',
        cmap='jet', vmin=0, vmax=2, alpha=0.6
    )
    
    # Title and labels
    title = ax.set_title("Animation of pollution difference from \n Gas Stove Max emission across Gallatin Valley")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # Create a colorbar
    colorbar = fig.colorbar(poll_img, ax=ax, label="% Difference from Max Concentration")
    
    # Update function for animation
    def update(frame):
        t_day = frame * dt * plot_freq
        poll_img.set_data(C_avgpertime[frame].T)
    
        # Interpolate wind info just like in main loop for the title
        day_idx = min(int(np.floor(t_day)), len(wind_mean) - 2)
        frac = t_day - int(np.floor(t_day))
        w_spd = (1 - frac) * wind_mean[day_idx] + frac * wind_mean[day_idx + 1]
        w_dir = (1 - frac) * wind_dir[day_idx] + frac * wind_dir[day_idx + 1]
    
        title.set_text(f"Day = {t_day:.1f}, Wind = {w_spd:.2f} mph, {w_dir:.0f}°")
    
        return [poll_img]
    
    # Only use every plot_freq-th frame
    frame_indices = list(range(len(Chistory)))
    
    ani = animation.FuncAnimation(
        fig, update, frames=frame_indices,
        interval=50, blit=False, repeat=False
    )
    
    plt.show()
        
    end_TA = time.perf_counter() # Starting the timer
    time_elapsed_TA = end_TA-start_TA
    
    print(f"the elapsed time is {time_elapsed_TA:0.2f}")
    
    return

#%% Check Criteria 1 and 2 for Belgrade, Bozeman, and Four Corners (Hayden Analysis)

def haydenanalysis(Chistory, steps, steps_per_day, Nx, Ny, Ndays, rescale):
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    C_Bel = Chistory[:, 13*rescale, 10*rescale]
    C_Bel_mean = np.mean(C_Bel)
    
    print(f"Average yearly Belgrade NO2 emissions = {C_Bel_mean:0.2f} ppb")
    
    # Criteria 2
    C_Bel_hourly = C_hourly[:, :, 13*rescale, 10*rescale]
    C_Bel_daily_max = np.max(C_Bel_hourly, axis = 1)
    C_Bel_p98 = np.percentile(C_Bel_daily_max, 98, axis=0)
    
    print(f"Belgrade p98 concentration = {C_Bel_p98:0.2f} ppb")
    
    
    ## Bozeman
    # Criteria 1
    C_Boz = Chistory[:, 17*rescale, 7*rescale]
    C_Boz_mean = np.mean(C_Boz)
    
    print(f"Average yearly Bozeman NO2 emissions = {C_Boz_mean:0.2f} ppb")
    
    # Criteria 2
    C_Boz_hourly = C_hourly[:, :, 17*rescale, 7*rescale]
    C_Boz_daily_max = np.max(C_Boz_hourly, axis = 1)
    C_Boz_p98 = np.percentile(C_Boz_daily_max, 98, axis=0)
    
    print(f"Bozeman p98 concentration = {C_Boz_p98:0.2f} ppb")
    
    
    ## Four Corners
    # Criteria 1
    C_FC = Chistory[:, 13*rescale, 7*rescale]
    C_FC_mean = np.mean(C_FC)
    
    print(f"Average yearly Four Corners NO2 emissions = {C_FC_mean:0.2f} ppb")
    
    # Criteria 2
    C_FC_hourly = C_hourly[:, :, 13*rescale, 7*rescale]
    C_FC_daily_max = np.max(C_FC_hourly, axis = 1)
    C_FC_p98 = np.percentile(C_FC_daily_max, 98, axis=0)
    
    print(f"Four Corners p98 concentration = {C_FC_p98:0.2f} ppb")
    
    
    ## Bridger
    # Criteria 1
    C_Brid = Chistory[:, 20*rescale, 12*rescale]
    C_Brid_mean = np.mean(C_Brid)
    
    print(f"Average yearly Bridger NO2 emissions = {C_Brid_mean:0.2f} ppb")
    
    # Criteria 2
    C_Brid_hourly = C_hourly[:, :, 20*rescale, 12*rescale]
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
    
    return

#%% leo analysis



#%% carson analysis