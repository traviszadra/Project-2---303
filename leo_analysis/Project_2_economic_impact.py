# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:04:03 2026

@author: Leoooo
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

#%% get data from state provided excel (from: https://lmi.mt.gov/LocalAreaProfiles)

data = pd.read_excel("Wage Growth Chart.xlsx", header = [0,1])
boz_manufacturing_wage = np.nanmax(data["Manufacturing", "2024"].values)

semi_production_wage = 94824 # from https://www.semiconductors.org/wp-content/uploads/2021/05/SIA-Impact_May2021-FINAL-May-19-2021_2.pdf
semi_eng_wage = 226145 

# hand copied data from pdf sources

boz_eng_wage = 84750 # from https://lmi.mt.gov/_docs/Publications/LMI-Pubs/Labor-Market-Publications/OEWS-Pub-2024-060425.pdf
boz_mech_wage = 95360
boz_elec_wage = 101920
boz_env_wage = 101410 # *data not available for bozeman, state average shown
boz_HSE_wage = 97920
boz_ind_wage = 90920
boz_mat_wage = 88910 # *state avg
boz_civil_wage = 97720
boz_chem_wage = 115380 # *state avg
boz_bio_wage = 105380 # *state avg
boz_CPUHW_wage = 139230 # *state avg

# array to make graphing easier


boz_eng_jobs = np.array([boz_eng_wage, boz_mech_wage, boz_elec_wage, boz_env_wage, boz_HSE_wage, boz_ind_wage, 
                      boz_mat_wage, boz_civil_wage, boz_chem_wage, boz_bio_wage, boz_CPUHW_wage])


all_eng_jobs_labels = np.array(["Semiconductor\nAverage","\"Architects\n& Engineers\"", "Mechanical", "Electrical", "Environmental",
 "Health and\nSafety", "Industrial", "Materials", "Civil", "Chemical", "Biological", "Computer\nHardware"])


is_local_data = np.array([False, False, False, False, True, False, False, True, False, True, True, True]) 



#%% add label to bar chart function adapted from geeksforgeeks https://www.geeksforgeeks.org/python/adding-value-labels-on-a-matplotlib-bar-chart/
def add_labels(x, y, y_label):
    
    y_offset = max(y) * .01 # make em float just a lil    
    
    for i in range(len(x)):
        plt.text(x[i], y[i]+y_offset, y_label[i], ha='center', fontsize=9)  # Aligning text at center
        
#%% givens

direct_jobs = 1200
induced_jobs = 4000    # NOTE: this is not specified as indirect or induced, but based on national averages,
                        # it is inline with induced, so that's what I'm assuming

# bozeman info (data from 2025/2024 - most recent available)
bozeman_pop = 61662          # https://weblink.bozeman.net/WebLink/DocView.aspx?id=313330&dbid=0&repo=BOZEMAN&searchid=bde67117-99a5-4b98-9db0-a596bcfd1678
gallatin_co_pop = 135227
bozeman_growth = 1681        # per year on average since 2010, slowing though
gallatin_co_manu_jobs = 3579 # 31 jobs lost in 2024
gallatin_co_tech_jobs = 843  # "professional and technical services"
prof_avg_wage = 119113       # pro and tech services as above ^ 9.6% increase from 2023 

# Bozeman workforce (age 25+) education stats (unused)

no_highschool = 0.03         # "high school or less, no diploma"
highschool = 0.132           
some_college = 0.138
associate_degree = 0.063
bachelors_degree = 0.386
graduate_degree = 0.251

bach_or_more = bachelors_degree+graduate_degree

gallatin_w_bach = bach_or_more * gallatin_co_pop
gallatin_wout_bach = (1-bach_or_more) * gallatin_co_pop

# national givens from https://www.semiconductors.org/chipping-in-sia-jobs-report/
ndj = 277000 #national direct jobs
nidj = 669498
n_inducedj = 905551

ndi = 47.1 *10**9 # national direct income
nii = 61.9 * 10**9
n_inducedi = 51.8 * 10**9

# national ratios

n_indirect_to_direct_jobs = nidj / ndj                # 2.4169
n_induced_to_direct_jobs = n_inducedj / ndj           # 3.269

n_indirect_to_direct_income = nii / ndi               # 1.314
n_induced_to_direct_income = n_inducedi / nii         # 0.8368

direct_income_to_jobs = ndi / ndj # income per job    $170,036!!!
indirect_income_to_jobs = nii / nidj                  # $92,457
induced_income_to_jobs = n_inducedi / n_inducedj      # $57,202

print("Avg incomes per job:")
print(f"direct: ${direct_income_to_jobs:.2f}")
print(f"indirect: ${indirect_income_to_jobs:.2f}")
print(f"induced: ${induced_income_to_jobs:.2f}\n")

# local ratio
l_induced_to_direct_jobs = induced_jobs/direct_jobs # aligns almost exactly with national average

#%% local calculations based on national ratios

indirect_jobs = direct_jobs * n_indirect_to_direct_jobs

print(f'indirect jobs: {indirect_jobs:.0f}\n')

direct_income = direct_jobs * direct_income_to_jobs
indirect_income = indirect_jobs * indirect_income_to_jobs
induced_income = induced_jobs * induced_income_to_jobs

print(f"direct income: ${direct_income:.2f}")         # $204,043,321.30
print(f"indirect income: ${indirect_income:.2f}")     # $369,829,334.82
print(f"induced income: ${induced_income:.2f}\n")     # $20,997,372.16

total_income = direct_income + indirect_income + induced_income
total_jobs = direct_jobs+indirect_jobs+induced_jobs

print(f"total income: ${total_income:.2f}")           # 798,276,988.24
print(f"total jobs: {total_jobs:.0f}")                # 9123

#%% graphing general jobs and income stats

jobs_labels= [f"Direct Jobs\n{direct_jobs:,}", f"Indirect Jobs\n{indirect_jobs:,.0f}", f"Induced Jobs\n{induced_jobs:,}"]           # I used claude ai to remind me how to set up a bar chart with labels on the x axis instead of numbers
jobs_ticks = np.arange(len(jobs_labels))
jobs_numbers = np.array([direct_jobs, indirect_jobs, induced_jobs])
avg_income = np.array([f"Avg yearly wages:\n${direct_income_to_jobs:,.0f}", f"Avg yearly wages:\n${indirect_income_to_jobs:,.0f}", f"Avg yearly wages:\n${induced_income_to_jobs:,.0f}"])

    

income_labels = [f"Direct Income\n${direct_income:,.0f}", f"Indirect Income\n${indirect_income:,.0f}", f"Induced Income\n${induced_income:,.0f}"]
income_ticks = np.arange(len(income_labels))
income_numbers = np.array([direct_income, indirect_income, induced_income]) / 10**6 # millions of dollars


plt.figure(1, figsize = (6,8))
plt.subplot(2,1,1)
plt.bar(jobs_ticks, jobs_numbers, label = f"Total Jobs: {total_jobs:,.0f}")
plt.xticks(jobs_ticks,labels = jobs_labels)
plt.ylabel("Number of jobs")
plt.title("Local Jobs Brought by SemiConducts")
plt.text(-.4, 3750, f"Total Jobs: {total_jobs:,.0f}", bbox=dict(boxstyle = "round", facecolor = 'white'))


plt.subplot(2,1,2)
plt.bar(income_ticks, income_numbers, label = income_numbers)
plt.xticks(income_ticks,labels = income_labels)
plt.ylabel("Income (Millions of Dollars)")
plt.title("Local Income Brought by SemiConducts")
plt.text(-.4, 260, f"Total Income:\n${(total_income/10**6):,.0f} million", bbox=dict(boxstyle = "round", facecolor = 'white'))
plt.subplots_adjust(hspace = 0.25)
add_labels(income_ticks, income_numbers, avg_income)
plt.ylim(0,299)


plt.show()

#%% graphing manufacturing/engineering data 

plt.figure(2, figsize = (16,9))

wage_width = 0.4

boz_wages = np.array((boz_manufacturing_wage, boz_eng_wage))
semi_wages = np.array((semi_production_wage, semi_eng_wage))

wage_labels = ["Manufacturing", "Architecture & Engineering"]

wage_ticks = np.arange(len(boz_wages))

plt.bar(wage_ticks+0.5*wage_width, boz_wages, width = wage_width, label = "Current Bozeman Average [8][9]")
plt.bar(wage_ticks -0.5* wage_width, semi_wages, width = wage_width, label = "Semiconductor Manufacturer\nNational Average[6]")
plt.xticks(wage_ticks, labels = wage_labels, fontsize = 20)
plt.ylabel("Yearly Wages ($)", fontsize = 20)
plt.legend(loc = "upper left", fontsize = 18)
plt.ylim(0,249999)
plt.title("Current Bozeman Wages Compared to Projected SemiConducts Wages", fontsize = 20)

for i in range(len(wage_ticks)):                                     # adapted from the geeksforgeeks function at the top
    y_offset = max(boz_wages) * .01 # make em float just a lil    
    plt.text(wage_ticks[i]+0.2, boz_wages[i]+y_offset, f"${boz_wages[i]:,.0f}", ha='center', fontsize=20)  # Aligning text at center
    plt.text(wage_ticks[i]-0.2, semi_wages[i]+y_offset, f"${semi_wages[i]:,.0f}", ha='center', fontsize=20)  # Aligning text at center

plt.show()

#%% graphing engineering data

plt.figure(3, figsize = (16,9))
# plt.figure(3)

boz_eng_jobs_ticks = np.arange(1, len(boz_eng_jobs)+1)
plt.bar(boz_eng_jobs_ticks, boz_eng_jobs, label = "Current Bozeman Average[9]")
plt.bar(0, semi_eng_wage, label = "Semiconductor Manufacturer\nNational Average[6]")

all_ticks = np.arange(0,len(boz_eng_jobs) + 1)

plt.xticks(all_ticks, labels = all_eng_jobs_labels)
plt.xlabel("Engineering Fields", fontsize = 20)
plt.ylabel("Yearly Wages ($)", fontsize = 20)
plt.title("Local Engineering Wages Compared to Projected SemiConducts Wages", fontsize = 30)

y_offset = semi_eng_wage * .01 # make em float just a lil    
    
x = boz_eng_jobs_ticks
y = boz_eng_jobs
y_label = boz_eng_jobs

# add asterisks if the data is actually statewide, not local

for i in range(len(x)):
    if is_local_data[i] == True:    
        plt.text(x[i], y[i]+y_offset, f"${y_label[i]:,}*", ha='center', fontsize=10)  # Aligning text at center
    else:
        plt.text(x[i], y[i]+y_offset, f"${y_label[i]:,}", ha='center', fontsize=10)  # Aligning text at center

plt.text(0, semi_eng_wage +y_offset, f"${semi_eng_wage:,}", ha='center', fontsize=10)  # Aligning text at center

handles, labels = plt.gca().get_legend_handles_labels()
custom = Line2D([0], [0], marker='*', color='black', linestyle='None', markersize=10, label='Local Data Unavailable, State Average Used') # got help from claude.ai with custom legend
handles.append(custom)
labels.append("Local Data Unavailable, State Average Used")
plt.legend(handles=handles, labels=labels, fontsize = 20)

#%% i used this section for notes/brainstorming

''' possible routes:
     note induced jobs current data for galltin county:
         determine which markets count (hospitality, retail, etc)
         compare to incoming induced jobs and salaries
     graph data:
         compare incoming jobs to current jobs
         compare incoming salaries to current salaries
         compare totals     
         
         
'''
