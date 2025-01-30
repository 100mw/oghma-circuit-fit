# Import modules python3.12.2

from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


model_version = 'Model V21-9'  

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "/Users/alexiarango/Library/CloudStorage/OneDrive-Personal/Documents/Oghma/Circuit/v21-4"
#sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\v21-9"
sim_folder_directory = "/Users/alexiarango/Documents/Oghma/v21-9"

# select device from dictionary with argument when the script is run in the terminal

device_dict = {1 : '30nm_d10_up',
               2 : '30nm_d10_down',
               3 : '40nm_d14_up',
               4 : '40nm_d14_down',
               7 : '56nm_d10_up',
               8 : '56nm_d10_down',
               5 : '60nm_d14_up',
               6 : '60nm_d14_down',
               9 : 'pause\\v19-4 c60 30nm d23 pause up'
               }
 
# save as .pdf or .png?
plot_file_type = '.png'






'''
Use command line positional arguments to select device number
'''

# Set up the argument parser
parser = argparse.ArgumentParser(description="Set device number based on the argument provided.")

# Add a positional argument
parser.add_argument(
    "device",  # The name of the positional argument
    type=int,      # Argument type
    help="Specify the device number."
)

# Parse the arguments
args = parser.parse_args()

# Set the variable to the provided value
device = args.device

# Print the result
print(f"The device is set to: {device}")






'''
Select elements to plot and set plotting attributes below
'''

# List names of circuit elements to plot:

elements = {
#  'Rs':    {'color': '#c10000', 'linewidth': 1, 'linestyle': '-',  'label': 'Series R'}, 
  'Rc':    {'color': '#fdfd04', 'linewidth': 1, 'linestyle': '-',  'label': 'Contact leg'},
#  'Rcsh':  {'color': '#fdfd04', 'linewidth': 1, 'linestyle': '-.', 'label': 'Contact Rcsh'},
  'Sc':    {'color': '#fdfd04', 'linewidth': 1, 'linestyle': ':',  'label': 'Contact Sc'},
  'St':    {'color': '#ff0000', 'linewidth': 1, 'linestyle': ':',  'label': 'Transport St'},
  'Tt':    {'color': '#ff0000', 'linewidth': 1, 'linestyle': '--', 'label': 'Transport Tt'},
  'Ii':    {'color': '#ff8400', 'linewidth': 1, 'linestyle': ':',  'label': 'Interface Ii'},
  'Di':    {'color': '#ff8400', 'linewidth': 1, 'linestyle': '--', 'label': 'Interface Di'}, 
  'Rsh':   {'color': '#91ff00', 'linewidth': 1, 'linestyle': '-',  'label': 'Shunt leg'},
  'Ssh':   {'color': '#437601', 'linewidth': 1, 'linestyle': ':',  'label': 'Shunt Ssh'},
  'Tsh':   {'color': '#437601', 'linewidth': 1, 'linestyle': '--', 'label': 'Shunt Tsh'},
  'Rshsh': {'color': '#437601', 'linewidth': 1, 'linestyle': '-.', 'label': 'Shunt Rshsh'}, 
  'Rb':    {'color': '#00e1ff', 'linewidth': 1, 'linestyle': '-',  'label': 'Barrier leg'}, 
  'Rbsh':  {'color': '#008394', 'linewidth': 1, 'linestyle': '-.', 'label': 'Barrier Rbsh'}, 
  'Rbdsh': {'color': '#008394', 'linewidth': 1, 'linestyle': '--', 'label': 'Barrier Rbdsh'}, 
  'Sb':    {'color': '#008394', 'linewidth': 1, 'linestyle': ':',  'label': 'Barrier SCLC'}
}

fit_name = 'jv'

calculate_mobility = True

# if above is True, list device specific values
epsilon_C60 = 2.3
thickness_MoO3 = 10
Nt = 5e23

# Set plot display properties
black_background = True
line_color = 'white'
plot_slopes = True
write_model_version = True
noise_threshold = 0.4
plot_width = 7
plot_height = 9
dpi = 120

# edit raw data display properties
point_color = '#ffffff'
point_size = 0
fit_color = '#df12d4'
fit_linewidth = 2
error_color = '#2d2d2d'

# Device area [m^2]
Area = 0.00000121	

q = 1.6e-19
epsilon0 = 8.854e-12

# Save fit parameter values?
save = True

# Set the display option to show significant digits
sig_dig = 3
pd.set_option('display.float_format', lambda x: f'%.{sig_dig}g' % x)







'''
Open raw jv data and fit data file, create numpy arrays
'''
# region

# create path to sim folder
sim_folder_path = Path(sim_folder_directory) / device_dict[device]

# open fit_data file containing raw data
data_raw = pd.read_csv(sim_folder_path / "fit_data0.inp", sep=r"\s+", comment="#", header=None)
data_raw = data_raw.loc[(data_raw!=0).any(axis=1)]

# create numpy arrays from raw data
v_data = data_raw[data_raw.columns[0]].to_numpy()
j_data = data_raw[data_raw.columns[1]].to_numpy()

# Eliminate first data point if errant
if v_data[0] < 1e-10:
  v_data = np.delete(v_data, [0])
  j_data = np.delete(j_data, [0])

# open file containing fit data
fit = pd.read_csv(sim_folder_path / 'sim' / fit_name / "jv.best", sep=r"\s+", comment="#", header=None)
fit = fit.loc[(fit!=0).any(axis=1)]

# create numpy arrays from fit data
v_fit = fit[fit.columns[0]].to_numpy()
j_fit = fit[fit.columns[1]].to_numpy()

# open file containing error data
error = pd.read_csv(sim_folder_path / 'sim' / fit_name / "fit_error_delta.csv", sep=r"\s+", comment="#", header=None)
error = error.loc[(error!=0).any(axis=1)]

# create numpy arrays from fit data
v_error = error[error.columns[0]].to_numpy()
j_error = error[error.columns[1]].to_numpy()

# endregion



''''
Extract current and voltage drops across circuit elements from netlist folders
'''
# region 

# Generate list of folder paths in netlist folder
netlist = sim_folder_path / "netlist/"
folders = list(netlist.glob('*/'))


# Iterate through each folder and read circuit_labels.dat files

#create empty DataFrame
data_df = pd.DataFrame()

#iterate over list with length = number of folders
for x in range(len(folders)):

  #get folder name, step number
  step = int(folders[x].name)
    
  #do not use .DS_Store file
  if step == ".DS_Store" or step == "results" or step == "data.json":
      continue

  temp = pd.DataFrame()

  #Read circuit_labels into DataFrame
  temp = pd.read_json(folders[x] / "circuit_labels.dat")

  #open data.json file
  with open(folders[x] / "data.json", 'r') as f:
    folder_data = json.load(f)

  # get voltage from data.json
  voltage = folder_data['voltage']

  # create list of voltages
  voltage_list = [voltage] * len(temp.columns)
  
  # append new voltage row to temp
  temp.loc['voltage'] = voltage_list

  #append new rows to DataFrame
  data_df = pd.concat([data_df, temp])

# Transpose DataFrame
#data_df.sort_index(axis=0,kind='stable',inplace=True)

# drop extra segments column
data_df = data_df.drop(columns=['segments'])
#data_df = data_df.drop(columns=['icons'])

# endregion



'''
Determine circuit element titles based on uid and replace segment column titles in data_df.
'''
# region

# find uid data_df
uid=data_df.iloc[1].to_list()

# turn circuit_labels uid into sim.json segment#
segments=[]
for x in uid:
	segments.append(f"segment{x}")

# read sim.json
myfile = open(sim_folder_path / 'sim.json')
sim = json.load(myfile)
myfile.close()

# grab "circuit_diagram" dictionary
circuit_diagram = sim['circuit']["circuit_diagram"]
del circuit_diagram['segments']

# find circuit element names 
titles=[]
for x in segments:
   titles.append(circuit_diagram[x]['name'])
#   print(x, circuit_diagram[x]['name'])
   
# rename dataframe column titles
for i, column in enumerate(data_df.columns):
	data_df = data_df.rename(columns={column: titles[i]})
   


pd.set_option('display.max_columns', None)

# to see a list of uid and element titles, print data_df
#print(data_df)

# endregion



'''
Read circuit element parameters from circuit_diagram dictionary
'''
# region

# create empty dataframes
resistors_df = pd.DataFrame(columns = ['R [Ohms]'])
powers_df = pd.DataFrame(columns = ['I0', 'Power'])
diodes_df = pd.DataFrame(columns = ['I0', 'n'])
barriers_df = pd.DataFrame(columns = ['I0', 'b0'])

for x in circuit_diagram:

  # find resistors
  if circuit_diagram[x]['comp'] == 'resistor':
     name = circuit_diagram[x]['name']
     resistance = circuit_diagram[x]['R']
     resistors_df.loc[name] = resistance

  # find power law elements   
  elif circuit_diagram[x]['comp'] == 'power':
     name = circuit_diagram[x]['name']
     I0 = circuit_diagram[x]['I0']
     power = circuit_diagram[x]['c']
     powers_df.loc[name] = [I0, power]

  # find diode elements   
  elif circuit_diagram[x]['comp'] == 'diode':
     name = circuit_diagram[x]['name']
     I0 = circuit_diagram[x]['I0']
     nid = circuit_diagram[x]['nid']
     diodes_df.loc[name] = [I0, nid]

  # find barrier elements   
  elif circuit_diagram[x]['comp'] == 'barrier':
     name = circuit_diagram[x]['name']
     I0 = circuit_diagram[x]['I0']
     b0 = circuit_diagram[x]['b0']
     barriers_df.loc[name] = [I0, b0]

# endregion



'''
Determine fit min/max values and add to element dataframes
'''
# region

# grab 'fit' dictionary from sim.json
fit_dict = sim['fits']['vars']
del fit_dict['segments']
del fit_dict['icon_']

# add min max columns to element dataframes
resistors_df['Rmin'] = np.nan
resistors_df['Rmax'] = np.nan
resistors_df = resistors_df.reindex(columns=['Rmin', 'R [Ohms]', 'Rmax'])

powers_df['I0min'] = np.nan
powers_df['I0max'] = np.nan
powers_df['Powmin'] = np.nan
powers_df['Powmax'] = np.nan
powers_df = powers_df.reindex(columns=['I0min', 'I0', 'I0max', 'Powmin', 'Power', 'Powmax'])

diodes_df['I0min'] = np.nan
diodes_df['I0max'] = np.nan
diodes_df['nmin'] = np.nan
diodes_df['nmax'] = np.nan
diodes_df = diodes_df.reindex(columns=['I0min', 'I0', 'I0max', 'nmin', 'n', 'nmax'])

barriers_df['I0min'] = np.nan
barriers_df['I0max'] = np.nan
barriers_df['b0min'] = np.nan
barriers_df['b0max'] = np.nan
barriers_df = barriers_df.reindex(columns=['I0min', 'I0', 'I0max', 'b0min', 'b0', 'b0max'])



# find min/max fit values that correspond to elment uid and add to element dataframes
for x in fit_dict:

  for element in resistors_df.index:

    # get uid for circuit element from data_df and seach fit_dict for uid
    if str(data_df.at['uid', element].iloc[0]) in fit_dict[x]['json_var']:
      resistors_df.at[element, 'Rmin'] = fit_dict[x]['min']
      resistors_df.at[element, 'Rmax'] = fit_dict[x]['max']

  for element in powers_df.index:
    if (str(data_df.at['uid', element].iloc[0]) + '.I0') in fit_dict[x]['json_var']:
      powers_df.at[element, 'I0min'] = fit_dict[x]['min']
      powers_df.at[element, 'I0max'] = fit_dict[x]['max']
    if (str(data_df.at['uid', element].iloc[0]) + '.c') in fit_dict[x]['json_var']:
      powers_df.at[element, 'Powmin'] = fit_dict[x]['min']
      powers_df.at[element, 'Powmax'] = fit_dict[x]['max']

  for element in diodes_df.index:
    if (str(data_df.at['uid', element].iloc[0]) + '.I0') in fit_dict[x]['json_var']:
      diodes_df.at[element, 'I0min'] = fit_dict[x]['min']
      diodes_df.at[element, 'I0max'] = fit_dict[x]['max']
    if (str(data_df.at['uid', element].iloc[0]) + '.nid') in fit_dict[x]['json_var']:
      diodes_df.at[element, 'nmin'] = fit_dict[x]['min']
      diodes_df.at[element, 'nmax'] = fit_dict[x]['max']

  for element in barriers_df.index:
    if (str(data_df.at['uid', element].iloc[0]) + '.I0') in fit_dict[x]['json_var']:
      barriers_df.at[element, 'I0min'] = fit_dict[x]['min']
      barriers_df.at[element, 'I0max'] = fit_dict[x]['max']
    if (str(data_df.at['uid', element].iloc[0]) + '.b0') in fit_dict[x]['json_var']:
      barriers_df.at[element, 'b0min'] = fit_dict[x]['min']
      barriers_df.at[element, 'b0max'] = fit_dict[x]['max']






print()
print(resistors_df)
print()
print(powers_df)
print()
print(diodes_df)
print()
print(barriers_df)
print()

# endregion



'''
Calculate mobility from space charge element
'''
# region

if calculate_mobility == True:
   
  # Searh folder name for thickness
  folder_name = sim_folder_path.name
  match = re.search(r'(..)nm', folder_name)
  thickness_C60 = int(match.group(1))

  I0_mu = powers_df.loc['St', 'I0']
  mu = 8/9/epsilon_C60/epsilon0*np.power(thickness_C60*1e-9, 3)*np.power(I0_mu,2)/Area
  
  print("mobility =", f'{mu:.3g}', "m^2/Vs", '\n')

# endregion



'''
Set up ploting
'''
# region

# Create figure
if black_background == True:
  plt.style.use('dark_background')							#black background



if plot_slopes == True:
  fig, (axs0, axs1) = plt.subplots(2, 
                                   1, 
                                   sharex=True, 
                                   figsize=(plot_width, 
                                            plot_height), 
                                   gridspec_kw={'height_ratios':[1,1]}, 
                                   dpi=dpi)
  
  #adjust plot margins
  fig.subplots_adjust(top=0.97, right=0.92, bottom=0.055, left=0.11, hspace=0)
  
else:
  fig, axs0 = plt.subplots(1, 1, sharex=True, figsize=(plot_width, plot_height), dpi=dpi, layout='constrained')

# Set title of window
fig.canvas.manager.set_window_title(sim_folder_path.name + " " + str(i)) 

# Set title of plot
axs0.set_title(sim_folder_path.name)

# add second set of axes for error
axs3 = axs0.twinx()
axs4 = axs3.twiny()

# Set up iv plot
axs0.set_ylabel("Current density [A/m$^2$]")
axs0.set_xscale("log")
axs0.set_yscale("log")
axs3.set_ylabel('Error')

# scientific notation for error axis
axs3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# hide tick labels for error x-axis
axs4.tick_params(axis='x', which='both', bottom=False, top=False)
axs4.set_xticks([])

# Label axes
if plot_slopes == True:
  axs1.set_ylabel("Slope of log-log j-v curve")
  axs1.set_xlabel("Voltage [V]")

else:
    axs0.set_xlabel("Voltage [V]")


if write_model_version == True:
  
  #text box for model version
  axs0.text(0.98, 0.025, model_version, transform=axs0.transAxes, ha='right')		

  # open file containing total error
  with open(sim_folder_path / 'fiterror.dat', 'r') as file:
    error_total = float(file.read())

  #text box for error
  axs0.text(0.02, 0.025, 'Error = ' + str(error_total), transform=axs0.transAxes, ha='left')

print('error = ', error_total)
print()

# endregion



'''
Add data to plot
'''
# region


# add error data to iv plot
axs4.plot(v_error,j_error, linewidth=fit_linewidth, color=error_color, linestyle='-', zorder=1)

# find voltage
voltage = data_df.loc['voltage', data_df.columns[0]].values
voltage = np.sort(voltage)
voltage = np.array(voltage, dtype=float)

# add all fit lines to plot
for name in elements:
    current_df = data_df.loc['i', name]
    current = current_df.values / Area
    current = np.sort(current)
    axs0.plot(voltage,
              current,
              color = elements[name]['color'],
              linewidth = elements[name]['linewidth'], 
              linestyle = elements[name]['linestyle'],
              label = elements[name]['label']
              )

# Add raw data to iv plot
axs0.plot(v_data, j_data, linewidth=fit_linewidth, color=point_color, linestyle='-',  marker='o', markersize=point_size, label='data', zorder=2)

# Add fit data to iv plot
axs0.plot(v_fit, j_fit, linewidth=fit_linewidth, color=fit_color, linestyle='-', label='fit', zorder=2)

axs0.set_zorder(axs4.get_zorder()+1)
axs0.patch.set_visible(False)


# Add log-log slope to plot
if plot_slopes == True:
  
  # calculate slope of data
  j_data_slope = np.gradient(np.log10(j_data), 
                             np.log10(v_data), edge_order=2)
  
  # calculate slope of fit
  j_fit_slope = np.gradient(np.log10(j_fit), 
                             np.log10(v_fit), edge_order=2)

  # add slope of fits to plot
  for name in elements:
      current_df = data_df.loc['i', name]
      current = current_df.values / Area
      current = np.sort(current)
      # find log-log slope of current
      current_np = np.array(current, dtype=float)
      slope = np.gradient(np.log10(current_np), np.log10(voltage), edge_order=2)
      axs1.plot(voltage,
                slope,
                color = elements[name]['color'],
                linewidth = elements[name]['linewidth'], 
                linestyle = elements[name]['linestyle']
                )
  
  # plot slope raw data
  axs1.plot(v_data, j_data_slope, linewidth=fit_linewidth, color=point_color, linestyle='-', marker='o', markersize=point_size)
  
  # Add fit data to slope plot
  axs1.plot(v_fit, j_fit_slope, linewidth=fit_linewidth, color=fit_color, linestyle='-')
  
# endregion



'''
Set axis limits
'''
# region

# set error axis limits
axs4.set_xlim(v_error[0], v_error[-1])
axs3.set_ylim(0.5e-4, 0.5e-3)

axs0.set_xlim(v_data[0], v_data[-1])

axs0.set_ylim(			
                j_data.min()*0.1,			# grab min from data
                j_data.max()*2				# grab max from data
)

if plot_slopes == True:

  # Find index above noise threshold
  noise_threshold_index = min(range(len(v_data)), key=lambda i: abs(v_data[i] - noise_threshold))

  axs1.set_ylim(			
                0.5,
                np.max(j_fit_slope)*1.2			# grab max from data
  )


axs0.legend()


#input('Press Enter to close plot and save...\n')

# endregion



'''
Calculate S and T area fractions
'''
# region

if calculate_mobility == True:

  # shunt area fraction from space charge currents
  I0_Ssh = powers_df.loc['Ssh', 'I0']
  I0_Tsh = powers_df.loc['St', 'I0']
  (fraction_S) = (np.power(thickness_MoO3 / thickness_C60 , 3) * 
                  np.power(I0_Ssh / I0_Tsh, 2))

  print('S area fraction = 'f'{fraction_S:.{sig_dig}g}')


  # function for MH power term
  def power_term(l):
    return np.power((l + 1) / l, l) * np.power((l + 1) / (2*l + 1), l + 1)

  # shunt area fraction from trapped charge currents
  lsh = powers_df.loc['Tsh', 'Power']-1
  I0_Tsh = powers_df.loc['Tsh', 'I0']
  lt = powers_df.loc['Tt', 'Power']-1
  I0_Tt = powers_df.loc['Tt', 'I0']
  fraction_T = (np.power(thickness_MoO3 * 1e-9, 2*lsh + 1) / 
                np.power(thickness_C60  * 1e-9, 2*lt + 1) *
                np.power(I0_Tsh, lsh + 1) / 
                np.power(I0_Tt, lt + 1) *
                power_term(lsh) / 
                power_term(lt) *
                np.power(q * Nt / epsilon0 / epsilon_C60, lsh - lt))

  print('T area fraction = 'f'{fraction_T:.{sig_dig}g}')
  print()

# endregion


'''
Save parameters as dictionary and txt file, circuit legs
'''
if save == True:
  

  '''
  Save circuit element parameters as dictionary to results folder
  '''

  # convert each dataframe to a dictionary
  resistors_dict = resistors_df.to_dict()
  powers_dict = powers_df.to_dict()
  diodes_dict = diodes_df.to_dict()
  barriers_dict = barriers_df.to_dict()

  # Create a dictionary to store these dictionaries
  save_dict = {'resistors': resistors_dict, 
                    'powers': powers_dict, 
                    'diodes': diodes_dict,
                    'barriers': barriers_dict}

  # path of new folder for results
  resultsfolder_path = sim_folder_path.parent / (sim_folder_path.stem + ' results')
  
  # make new folder
  resultsfolder_path.mkdir(parents=True, exist_ok=True)

  # check if file exits
  filename = Path(resultsfolder_path) / 'Fit values .json'
  i = 1
  while (new_filename := filename.with_stem(f"{filename.stem}{i}")).exists():
    i += 1
  

  # Save dictionary to a JSON file
  with open(new_filename, 'w') as f:
    json.dump(save_dict, f)

  print('Parameters Saved', '\n')


  # save figure to directory
  figurepath = sim_folder_path.with_suffix(plot_file_type)

  plt.savefig(figurepath)

  print('Figure Saved', '\n')




  '''
  Save circuit element parameters as csv to results folder
  '''

  # add empty row at the end of each dataframe
  resistors_df.loc[len(resistors_df.index)] = None
  powers_df.loc[len(powers_df.index)] = None
  diodes_df.loc[len(diodes_df.index)] = None
  barriers_df.loc[len(barriers_df.index)] = None

  # concatenate dataframes
  save_df = pd.concat([resistors_df, 
                       powers_df, 
                       diodes_df,
                       barriers_df])

  # create filename
  filename = Path(resultsfolder_path) / (device_dict[device] + '.txt')

  # save dataframe as csv
  save_df.to_csv(filename, sep='\t', index=True)




  '''
  Add circuit legs to dataframe and save to results folder
  '''

  # empty datafram
  circuit_legs_df = pd.DataFrame()

  # add voltage
  circuit_legs_df['Voltage [V]'] = voltage

  # add circuit legs
  circuit_legs_df['Contact [A/m^2]'] = np.sort(data_df.loc['i', 'Rc'] / Area)
  circuit_legs_df['Shunt [A/m^2]'] = np.sort(data_df.loc['i', 'Rsh'] / Area / (fraction_S))
  circuit_legs_df['Barrier [A/m^2]'] = np.sort(data_df.loc['i', 'Rb'] / Area)

  # create path to file to save to
  filename = Path(resultsfolder_path) / 'Circuit_legs.txt'

  # save to results folder
  circuit_legs_df.to_csv(filename, sep='\t', index=False)

