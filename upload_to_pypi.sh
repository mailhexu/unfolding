#!/bin/bash

# Script to build and upload the package to PyPI

set -e  # Exit immediately if a command exits with a non-zero status

echo "Building the package..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

echo "Package built successfully!"

echo "Uploading to PyPI..."

# Check if we have the built package
if [ ! -d "dist" ]; then
    echo "Error: No dist directory found. Please build the package first."
    exit 1
fi

# Upload to PyPI using twine
python -m twine upload dist/*

echo "Package uploaded successfully!"