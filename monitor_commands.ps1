# Function to unregister all previous events related to the FileSystemWatcher
function Clear-EventHandlers {
    Get-EventSubscriber | Where-Object { $_.SourceObject -is [System.IO.FileSystemWatcher] } | Unregister-Event
}

# Clear any previous FileSystemWatcher events before starting
Clear-EventHandlers

# Get the directory where the script is located
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Display the initial monitoring message
Write-Host "Monitoring for commands from the Mac terminal"

# Create a FileSystemWatcher to monitor the directory and its subdirectories
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $scriptDirectory
$watcher.Filter = "trigger.txt"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Define the action to take when trigger.txt is created
$action = {
    # Use the automatic variable $Event.SourceEventArgs
    $filePath = $Event.SourceEventArgs.FullPath
    $directory = Split-Path $filePath -Parent
    Write-Host "trigger.txt created in $directory"

    # Change to the directory where trigger.txt was found
    Push-Location $directory
    try {
        # Run the first command and wait for it to finish
        Start-Process "C:\Program Files (x86)\OghmaNano\oghma_core.exe" -ArgumentList "--1fit" -Wait -NoNewWindow -WorkingDirectory $directory

        # Run the second command and wait for it to finish
        Start-Process "C:\Program Files (x86)\OghmaNano\oghma_core.exe" -Wait -NoNewWindow -WorkingDirectory $directory
    } finally {
        # Return to the original directory
        Pop-Location
    }

    # Delete the trigger.txt file
    Remove-Item -Path $filePath -Force
    Write-Host "Deleted trigger.txt in $directory"

    # Display the monitoring message again
    Write-Host "Monitoring for commands from the Mac terminal"
}

# Register the event handler for the Created event
Register-ObjectEvent -InputObject $watcher -EventName Created -Action $action

# Display the initial monitoring message
Write-Host "Monitoring for commands from the Mac terminal"

# Keep the script running indefinitely
while ($true) {
    Start-Sleep -Seconds 1
}
