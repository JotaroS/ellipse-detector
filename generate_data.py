# code for genrating cropped images with an ellipse, then .json file that contains ellipse properties.

import cv2
import random
import glob
import json
import numpy as np

DEBUG = True
ITERATIONS = 5
CROP_SIZE = 128

def preview_image(path):
    image = cv2.imread(path)
    cv2.imshow("image window", image)
    return

def crop_image(path):
    src = cv2.imread(path)
    grayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if (grayscale.shape[0] > CROP_SIZE) and (grayscale.shape[1]> CROP_SIZE):
        cropped = grayscale[0:CROP_SIZE, 0:CROP_SIZE] #TODO: properly randomize coordinates to crop.
        # previewing processed images
        # cv2.imshow('gray scale', grayscale)
        # cv2.imshow('src',src)
        # cv2.imshow('cropped', cropped)
        return cropped
    print("[cv] skipping images as it was not larger than 128")
    return None

def gen_random_noise_image():
    src = np.random.randint(0, 256, (CROP_SIZE, CROP_SIZE, 3), np.uint8)
    ret = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    return ret

def draw_ellipse(image):
    # color / thickness
    brightness = random.randint(0, 10)
    thickness  = 0
    center     = (int(CROP_SIZE/2) + random.randint(-50, 50), int(CROP_SIZE/2) + random.randint(-50, 50))
    radius     = random.randint(10, 20)
    axisLength = (radius, radius)
    angle      = 0
    color      = (brightness, brightness, brightness)
    
    drawn = cv2.ellipse(image, center, axisLength, angle, 0, 360, color, -1)
    
    ellipse_data = {
        'center': center,
        'radius': radius
    }
    # cv2.imshow("ellipse", drawn)
    return drawn, ellipse_data

def process_image(path):
    cropped              = crop_image(path)
    drawn, ellipse_data  = draw_ellipse(cropped)
    return drawn, ellipse_data

def main():
    # process all data on 'images' folder
    files = glob.glob("images/*.jpg")
    files.sort()
    metadata = []
    file_idx = 0;
    for f in files:
        for k in range(0, ITERATIONS):
            img_name = f.split('\\')[1]
            img, ellipse_data = process_image(f)
            out_filename = str(file_idx).zfill(10)
            try:
                cv2.imwrite('./cropped/'+out_filename+".jpg", img)
                metadata.append({
                    'img_name': out_filename+".jpg",
                    'center'  : ellipse_data['center'],
                    'radius'  : ellipse_data['radius']
                })
                file_idx += 1
                print(out_filename+".jpg", 'has been exported.')
            except:
                print(f, 'was not processed properly. skipping...')

    with open ('labels.json','w') as outfile:
        json.dump(metadata, outfile)

    return;

if __name__ == '__main__':
    main()