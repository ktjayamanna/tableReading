#!/bin/bash

# Define the image path variable
image_path="data/sample_table.bmp"
# image_path="data/data/Marmot_data/10.1.1.1.2043_4.bmp"

# Run the Python script with the image path variable
python plot_masks.py --image_path="$image_path"
