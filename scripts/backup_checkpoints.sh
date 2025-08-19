#!/bin/bash

# Backup script for checkpoints
# Run via crontab -e

# Set the source and destination directories
SOURCE_DIR="/checkpoints"
DEST_DIR="$HOME/checkpoints_backup"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"
LOG_FILE="$DEST_DIR/backup.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    log_message "ERROR: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check if source directory is empty
if [ -z "$(ls -A "$SOURCE_DIR" 2>/dev/null)" ]; then
    log_message "WARNING: Source directory $SOURCE_DIR is empty"
    exit 0
fi

log_message "Starting checkpoint backup from $SOURCE_DIR to $DEST_DIR"

# Copy all checkpoints with progress, do not overwrite existing files
if rsync -av --ignore-existing --progress "$SOURCE_DIR/" "$DEST_DIR/"; then
    log_message "SUCCESS: Checkpoint backup completed successfully"
else
    log_message "ERROR: Checkpoint backup failed"
    exit 1
fi

log_message "Backup process completed" 
