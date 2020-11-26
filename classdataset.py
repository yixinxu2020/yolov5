import random
import pandas as pd
from shutil import copy2

# devide the train_dataset to train & val as the ratio of 9:1
df = pd.read_csv('img_df.csv')
img_name = df['img_name']
img_name = list(img_name)
txt_name = []
for name in img_name:
    name = name.replace('png', 'txt')
    txt_name.append(name)
index = list(range(len(img_name)))
random.shuffle(index)
train_index = index[0:int(0.9 * len(index))]
val_index = index[int(0.9 * len(index)):]


def devide_dataset(img_source_path, labels_source_path, new_dir):
    for phase in ['images', 'labels']:
        if phase == 'images':
            file_name = img_name
            source_path = img_source_path
        if phase == 'labels':
            file_name = txt_name
            source_path = labels_source_path

        for m in train_index:
            origins_file_path = source_path + file_name[m]
            new_train_file_path = new_dir + phase + '/train/' + file_name[m]
            copy2(origins_file_path, new_train_file_path)
        for n in val_index:
            origins_file_path = source_path + file_name[n]
            new_val_file_path = new_dir + phase + '/val/' + file_name[n]
            copy2(origins_file_path, new_val_file_path)


def main():
    devide_dataset('./data/train/', './train_txt/', './yolov5/digit/')


if __name__ == '__main__':
    main()
