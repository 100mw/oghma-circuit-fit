# oghma-circuit-fit
Python scripts to help with circuit modeling and fitting in OghmaNano

Procedure for running manual fits

1. Running jvslope.py will plot the log-log j-v curve and its slope. It creates a text file containing the circuit parameters and their max/min limits. 
2. The text file can be edited and saved with updated values. 
3. On the windows virtual machine, monitor_commands.ps1 must be running in the master directory where the OghmaNano simulation files are located. 
4. Then run setparams_vm.py to write the values to OghmaNano and update the plot of the log-log j-v curve and slope.

# Configuring scripts

Before running the scripts, the directory and list of devices must be edited in VS Code.

# Syntax

python3 jvslope.py {device number}

where {device number} is set in the list of devices at the beginning of the script.
