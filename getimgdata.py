import cv2
import pandas as pd

# use the img_bbox_data and images merge the all data, include
# img_name, label, left, top, width, height, bottom, right, img_width, img_height
df = pd.read_csv('img_bbox_data.csv')
img_list = list(df['img_name'])
img_name = []
for i, v in enumerate(img_list):
    if img_list.index(v) == i:
        img_name.append(v)
img_data = pd.DataFrame({'img_name': img_name})
img_dir = './data/train/'
width = []
height = []
for name in img_name:
    img_path = img_dir + name
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    width.append(w)
    height.append(h)

img_data['img_width'] = width
img_data['img_height'] = height

img_df = df.merge(img_data, on='img_name', how='left')
img_df.to_csv('img_alldata.csv')
