import os
import cv2
import numpy as np

img_size=(64,64)
min_images_per_person=3

def load_image(dataset_path):
    X=[] #dataset of all images in numerical form
    y=[]
    label_map={}
    label_counter=0
    
    for person in os.listdir(dataset_path):
        person_folder=os.path.join(dataset_path,person)
        if not os.path.isdir(person_folder):
            continue

        image_files=os.listdir(person_folder)
        if len(image_files)<min_images_per_person:
            continue

        if person not in label_map:
            label_map[person]=label_counter
            label_counter+=1

        label=label_map[person]

        # print("Person:", person)
        # print("Label:",label)

        for img_file in image_files:
            img_path=os.path.join(person_folder,img_file)
            img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img=cv2.resize(img, img_size)
            X.append(img.flatten()) #each iteration appends the pixel intensity of all pixels in an image [100,54,23,46....89]            y.append(label)
            y.append(label)
            
    return np.array(X), np.array(y),label_map