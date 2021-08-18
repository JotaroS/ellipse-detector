# code for genrating cropped images with an ellipse, then .json file that contains ellipse properties.

import cv2
import random
import glob
import json

DEBUG = True
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

def draw_ellipse(image):
    # color / thickness
    brightness = random.randint(0, 255)
    thickness  = 0
    center     = (int(CROP_SIZE/2) + random.randint(-20, 20), int(CROP_SIZE/2)  + random.randint(-20, 20))
    radius     = random.randint(5, 40)
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
    for f in files:
        img_name = f.split('/')[1]
        print(img_name)
        img, ellipse_data = process_image(f)
        try:
            cv2.imwrite('./cropped/'+img_name, img)
            metadata.append({
                'img_name': img_name,
                'center'  : ellipse_data['center'],
                'radius'  : ellipse_data['radius']
            })
        except:
            print(f, 'was not processed properly. skipping...')

    with open ('labels.json','w') as outfile:
        json.dump(metadata, outfile)

    return;

if __name__ == '__main__':
    main()