#!/bin/bash

# Script to clean all directories (and their content) from logs directories
# while preserving the logs directories themselves and their .gitignore files

set -e  # Exit on any error

# Function to clean a logs directory
clean_logs_dir() {
    local logs_dir="$1"

    if [ ! -d "$logs_dir" ]; then
        echo "Warning: Directory '$logs_dir' does not exist, skipping..."
        return
    fi

    echo "Cleaning directory: $logs_dir"

    # Find all directories in the logs directory and remove them
    # Use -mindepth 1 to avoid removing the logs directory itself
    # Use -maxdepth 1 to only remove direct subdirectories
    find "$logs_dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;

    # Count remaining items (should only be .gitignore if present)
    local remaining_count=$(find "$logs_dir" -mindepth 1 | wc -l)
    echo "  → Removed all subdirectories. Remaining files: $remaining_count"
}

echo "Starting logs cleanup..."
echo "=================="

# Clean the main logs directory
clean_logs_dir "logs"

# Clean the test logs directory
clean_logs_dir "test/logs"

echo "=================="
echo "Logs cleanup completed!"

# Optional: Show what's left in the directories
echo ""
echo "Contents after cleanup:"
echo "logs/:"
ls -la logs/ 2>/dev/null || echo "  Directory does not exist"
echo ""
echo "test/logs/:"
ls -la test/logs/ 2>/dev/null || echo "  Directory does not exist"