import os
import h5py
import pandas as pd


# read the digitStruct.mat, get the img_bbox_data
def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr[i].item()][0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs


def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file, 'r')
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([], columns=['height', 'img_name', 'label', 'left', 'top', 'width'])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict, orient='columns')])
    bbox_df['bottom'] = bbox_df['top'] + bbox_df['height']
    bbox_df['right'] = bbox_df['left'] + bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df


def construct_all_data(img_folder, mat_file_name, csv_name):
    img_bbox_data = img_boundingbox_data_constructor(os.path.join(img_folder, mat_file_name))
    img_bbox_data.to_csv(csv_name, index=False)


def main():
    train_folder = "./data/train"
    construct_all_data(train_folder, 'digitStruct.mat', 'img_bbox_data.csv')


if __name__ == '__main__':
    main()
