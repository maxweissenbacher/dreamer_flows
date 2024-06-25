#!/bin/bash

# Temporary file to store failed installations
LOG_FILE="failed_packages.log"

# Remove existing log file if exists
rm -f "$LOG_FILE"

# Read the requirements file line by line
while IFS= read -r package; do
    # Try installing the package, redirecting errors to the log file
    pip install "$package" || echo "$package" >> "$LOG_FILE"
done < requirements.txt

echo "Installation complete. Failed packages listed in '$LOG_FILE'."

