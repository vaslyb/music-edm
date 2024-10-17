import pandas as pd
import json
import os

NOTE_MAPPING = {
    'C': 0.0, 'C#': 1.0, 'D': 2.0, 'D#': 3.0, 'E': 4.0, 'F': 5.0, 'F#': 6.0, 'G': 7.0, 'G#': 8.0, 'A': 9.0, 'A#': 10.0, 'B': 11.0
}


def create_dataset():
    df = pd.DataFrame()
    header = list()
    data = list()
    header_labels = ['id']
    labels = list()
    label_list1 = [1,0,0,0]
    label_list2 = [0,1,0,0]
    label_list3 = [0,0,1,0]
    label_list4 = [0,0,0,1]
    for filename in os.listdir('../../results/features'):
        header_labels.append(filename)
        for file in os.listdir('../../results/features/'+filename):
            if file.endswith('.json'):
                #print("file",file)
                with open(f'../../results/features/{filename}/{file}', 'r') as f:
                    features = dict()
                    features['id'] = file.split('.')[0]
                    features.update(json.load(f))
                    if not header:
                        header = list(features.keys())
                    features['key'] = NOTE_MAPPING[features['key']]
                    features['chord'] = NOTE_MAPPING[features['chord']]
                    features['key_scale'] = 1.0 if features['key_scale'] == 'major' else 0.0
                    features['chord_scale'] = 1.0 if features['chord_scale'] == 'major' else 0.0
                    data.append(list(features.values()))
                    if len(header_labels) == 2:
                        labels.append([features['id']]+label_list1)
                    elif len(header_labels) == 3:
                        labels.append([features['id']]+label_list2)
                    elif len(header_labels) == 4:
                        labels.append([features['id']]+label_list3)
                    elif len(header_labels) == 5:
                        labels.append([features['id']]+label_list4)
    print(len(data), len(labels))
    df = pd.DataFrame(data, columns=header)
    os.makedirs('../../results/dataset', exist_ok=True)
    df.to_csv('../../results/dataset/data.csv', index=False)
    df = pd.DataFrame(labels, columns=header_labels)
    df.to_csv('../../results/dataset/labels.csv', index=False)
                
if __name__ == '__main__':
    create_dataset()
    