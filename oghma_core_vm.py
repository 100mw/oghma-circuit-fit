# rum ogmha_core.exe on windows virtual machine

from pathlib import Path
import argparse
import time

# Type path to folder here (Mac/Windows/Unix compatible):
#sim_folder_directory = "C:\\Users\\acara\\OneDrive\\Documents\\Oghma\\Circuit\\demo"
sim_folder_directory = "/Users/alexiarango/Documents/Oghma/demo"


# select device from dictionary if running from vscode
device = 1

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





'''
Use positional arguments to select device number
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
Execute commands on windows vm
'''

command = 'execute the command'

# create path to sim folder
sim_folder_path = Path(sim_folder_directory) / device_dict[device]

# Path to command.txt in the simulation directory
command_file = sim_folder_path / 'trigger.txt'

print(f"Sending command to {sim_folder_path}: {command}")

# Write the command to command.txt
with open(command_file, 'w') as f:
    f.write(command)

# Wait for the command file to be deleted by the PowerShell script
while command_file.exists():
    time.sleep(1)

print(f"Command '{command}' executed in {sim_folder_path}.")
