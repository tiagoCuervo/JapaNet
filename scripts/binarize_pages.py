import os
import pandas as pd
import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


dataPath = "./data"
train_df = pd.read_csv(os.path.join(dataPath, "train.csv").replace("\\","/"))
character_dict = pd.read_csv(os.path.join(dataPath, "unicode_translation.csv").replace("\\","/"))


class JapChar():
    def __init__(self, char_data, im_id):
        self.char = char_data[0]
        self.x = int(char_data[1])
        self.y = int(char_data[2])
        self.width = int(char_data[3])
        self.height = int(char_data[4])
        self.im_id = im_id

    def get_area(self):
        return self.width * self.height

    def get_file(self):
        return os.path.join(dataPath, f"train/{self.im_id}.jpg")

    def get_top_left(self):
        return [self.x, self.y]

    def get_bottom_right(self):
        return [self.x + self.width, self.y + self.height]

    def show(self):
        plt.figure(figsize=(6, 6))
        im = PIL.Image.open(self.get_file())
        im = im.crop(self.get_top_left() + self.get_bottom_right())
        plt.imshow(im)


class ScripturePage:
    def __init__(self, im_data):
        self.id = im_data[0]
        if type(im_data[1]) is not float:
            split_labels = im_data[1].split()
            self.labels = [JapChar(split_labels[i: i+5], self.id)
                           for i in range(0, len(split_labels), 5)]
        else:
            self.labels = []

    def get_file(self):
        return os.path.join(dataPath, f"train/{self.id}.jpg")

    def show(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(plt.imread(self.get_file()))

    def get_im(self):
        return PIL.Image.open(self.get_file())

    def show_labeled(self):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        plt.imshow(self.get_im())

        for label in self.labels:
            box = Rectangle((label.x, label.y), label.width,
                            label.height, fill=False, edgecolor='r')
            ax.add_patch(box)

        plt.show()


data = [ScripturePage(train_df.loc[i]) for i in range(len(train_df))]


def page_binarizer(data):
    '''
    taken from the following Kaggle notebook https://www.kaggle.com/pr1c3f1eld/data-cleaning-and-pre-processing
    '''

    img_0_orig = cv2.imread(data)
    img_0_orig = cv2.cvtColor(img_0_orig, cv2.COLOR_BGR2RGB)

    img_0 = cv2.imread(data)
    img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    
    # "Ben's prerocessing"
    blur = cv2.GaussianBlur(img_0,(3,3),0)
    sharp_mask = np.subtract(img_0, blur)
    img_0 = cv2.addWeighted(img_0,1, sharp_mask,10, 0)

    ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel_1 = np.ones((5,5),np.uint8)
    kernel_2 = np.ones((1,1),np.uint8)

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_2)

    mask = cv2.bitwise_not(closing)
    mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    img = cv2.add(img_0_orig,mask)

    # "Ben's prerocessing"
    blur_1 = cv2.GaussianBlur(img, (13,13), 0)
    sharp_mask_1 = np.subtract(img,blur_1)
    sharp_mask_1 = cv2.GaussianBlur(sharp_mask_1, (7,7), 0)
    img = cv2.addWeighted(img,1,sharp_mask_1,-10, 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_bw_pages(data):
    '''
    data: List of ScripturePage
    '''
    pages_checked = []
    print("Starting binarizing")
    os.makedirs(f"./data/bw_train", exist_ok=True)
    for page in tqdm(data):
        binary_page = page_binarizer(page.get_file())
        im_page = PIL.Image.fromarray(binary_page)
        id_page = page.id
        im_page.save(f"./data/bw_train/{id_page}.jpg")
        pages_checked.append(id_page)
    print("All pages checked!")
    return pages_checked

bw_pages = make_bw_pages(data)


''' 
WORK IN PROGRESS
def make_bw_classifier_data(data, crop_size=(64, 64)):

    pages_checked = []
    print("Starting cropping")
    for page in tqdm(data):
        # counter to enumerate characters
        counter = 0
        #open the respective image of the page
        bin_page = page_binarizer(page.get_file())
        im_page = PIL.Image.fromarray(bin_page)
        id_page = page.id
        pages_checked.append(id_page)
        for char in page.labels:
            # cropping the respective character
            im_char = im_page.crop(
                char.get_top_left() + char.get_bottom_right())
            # resizing
            im_char = im_char.resize(crop_size)
            # saving character image
            os.makedirs(f"./data/bw_train_char/{char.char}", exist_ok=True)
            im_char.save(f"./data/bw_train_char/{char.char}/{id_page}_{counter}.jpg")

            counter += 1
    print("All pages checked!")
    return pages_checked
'''