import pandas as pd

# transfer the test result to suitable .josn file
df = pd.read_json('/home/div/cv/hw2/yolov5/runs/test/exp21/best_predictions.json', lines=True)
df = df.T
df = pd.Series(df[0])
image_id = []
bbox = []
score = []
label = []
for dic in df:
    image_id.append(dic['image_id'])
    bbox.append(dic['bbox'])
    score.append(dic['score'])
    if dic['category_id'] == 0:
        label_id = 10
    else:
        label_id = dic['category_id']
    label.append(label_id)

dataframe = pd.DataFrame({
    'image_id': image_id,
    'bbox': bbox,
    'score': score,
    'label': label
})
dataframe.to_csv('pre_result.csv')
print('begin process bbox')
data = pd.read_csv('pre_result.csv')
bbox_ = []
for i in range(len(data['bbox'])):
    box = eval(data['bbox'][i])
    bbox_0 = [1, 1, 1, 1]
    bbox_0[0] = box[1]
    bbox_0[1] = box[0]
    bbox_0[2] = box[1] + box[3]
    bbox_0[3] = box[0] + box[2]
    bbox_.append(bbox_0)
data['bbox'] = bbox_
print(data['bbox'][0])
data = data.groupby('image_id')
print(data.size())
result = []
for i in range(13068):
    direction = {}
    a = data.get_group(i+1)
    direction['bbox'] = list(a['bbox'])
    direction['score'] = list(a['score'])
    direction['label'] = list(a['label'])
    result.append(direction)

result = pd.DataFrame(result)
result.to_json('result.json', orient='records')