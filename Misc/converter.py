#!/usr/bin/python3.6
from osgeo import gdal
import re
import os
import math

def convertfiles(path):
    #Converts jp2 images to TIFF.
    options_string = '-of JPEG -outsize 50% 50%'
    files = [path + "/" + file for file in os.listdir(path) if file.endswith(".jp2")]
    num_files = len(files)
    progress = 0
    processed = 0

    for file in files:

        # Print percentage completed when at least one percent have been processed.
        processed += 1
        if (math.floor(processed / num_files * 100) > progress):
            progress = math.floor(processed / num_files * 100)
            print(f"Processing: {progress} %")

        in_img = gdal.Open(file)
        gdal.Translate(re.sub('.jp2', '', file) + '.jpg', in_img, options=options_string)
        del in_img
