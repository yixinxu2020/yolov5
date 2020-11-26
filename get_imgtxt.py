import pandas as pd
import os


# get the images labels information saved as images_id.txt
# data-pre-process: form construct_dataset.py get img_bbox-data.csv, from getimgdata.py get img_alldata.csv,
# then use img_alldata get labels txt format with get_imgtxt.py
def get_imgtxt(img_folder, csv_name, txt_folder):
    df = pd.read_csv(csv_name)
    df = df.groupby('img_name')
    name_list = os.listdir(img_folder)
    for name in name_list:
        data = df.get_group(name)
        labels = []
        for i, label in enumerate(data['label']):
            if label == 10.0:
                label = 0
            else:
                label = int(label)
            labels.append(label)
        # normalize the bbox data for yolov5 format
        x_center = list((data['left'] + data['width'] * 0.5) / data['img_width'])
        y_center = list((data['top'] + data['height'] * 0.5) / data['img_height'])
        width = list(data['width'] / data['img_width'])
        height = list(data['height'] / data['img_height'])
        height_ = []
        for i, v in enumerate(height):
            if v > 1:
                print(name, v)
                v = 1
            height_.append(v)
        info = pd.DataFrame({
            'class': labels,
            'x_cenyer': x_center,
            'y_center': y_center,
            'width': width,
            'height': height_
        })
        name = name.replace('png', 'txt')
        path = txt_folder + name
        info.to_csv(path, sep='\t', index=False, header=0)


def main():
    get_imgtxt('./data/train', 'img_alldata.csv', './train_txt/')


if __name__ == '__main__':
    main()
