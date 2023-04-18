#!/bin/bash

# Set the maximum directory size in bytes
MAX_SIZE=$((1024 * 1024 * 1024)) # 1 GB

# Set the names of the directories to monitor and delete files from
DIR_NAME1="anhthe"
DIR_NAME2="invalidimg"

# Check if the first directory size exceeds the maximum
if [ $(du -sb $DIR_NAME1 | awk '{print $1}') -gt $MAX_SIZE ]; then
  # Delete all files in the first directory (excluding hidden files)
  find $DIR_NAME1 -maxdepth 1 ! -name ".*" -type f -delete
fi

# Check if the second directory size exceeds the maximum
if [ $(du -sb $DIR_NAME2 | awk '{print $1}') -gt $MAX_SIZE ]; then
  # Delete all files in the second directory (excluding hidden files)
  find $DIR_NAME2 -maxdepth 1 ! -name ".*" -type f -delete
fi