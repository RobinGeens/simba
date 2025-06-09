#!/bin/bash

STOP_FILE="/tmp/main_persistent.stop"

# Function to handle cleanup
cleanup() {
    echo "Cleaning up..."
    pkill -f "main.sh" 2>/dev/null
    rm -f "$STOP_FILE" 2>/dev/null
    echo "Script terminated."
    exit 0
}

# Remove stop file if it exists from a previous run
rm -f "$STOP_FILE"

while true; do
    # Check if stop file exists
    if [ -f "$STOP_FILE" ]; then
        echo "Stop file detected. Terminating..."
        cleanup
    fi

    ./main.sh
    if [ $? -eq 0 ]; then
        echo "main.sh completed successfully"
        break
    else
        echo "main.sh failed, restarting..."
        sleep 1  # Add a small delay before restarting
    fi
done
