import os
import pandas as pd
import PIL
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dataPath = "./data"
train_df = pd.read_csv(os.path.join(dataPath, "train.csv").replace("\\", "/"))
character_dict = pd.read_csv(os.path.join(dataPath, "unicode_translation.csv").replace("\\", "/"))


class JapChar:
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
            self.labels = [JapChar(split_labels[i: i + 5], self.id)
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


def make_classifier_data(data, crop_size=(64, 64)):
    '''
    data: List of ScripturePage
    '''
    pages_checked = []
    print("Starting cropping")
    for page in data:
        # counter to enumerate characters
        counter = 0
        # open the respective image of the page
        im_page = page.get_im()
        id_page = page.id
        pages_checked.append(id_page)
        for char in page.labels:
            # cropping the respective character
            im_char = im_page.crop(
                char.get_top_left() + char.get_bottom_right())
            # resizing
            im_char = im_char.resize(crop_size)
            # saving character image
            os.makedirs(f"./data/train_char/{char.char}", exist_ok=True)
            im_char.save(f"./data/train_char/{char.char}/{id_page}_{counter}.jpg")

            counter += 1
    print("All pages checked!")
    return pages_checked


cropped_pages = make_classifier_data(data)

# deleting extra characters not available in the unicode dictionary.
ALL_CLASSES = set(np.load("./data/ALL_CLASSES.npy"))
with open('./data/CLASSES', 'rb') as fp:
    CLASSES = pickle.load(fp)

img_classes_path = Path('./data/train_char')
CLASSES = set(CLASSES)

CLASSES_ERASE = ALL_CLASSES - CLASSES

for CLASS in CLASSES_ERASE:
    folder_path = os.path.join(img_classes_path, f"{CLASS}")
    try:
        folder_path.rmdir()
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")
