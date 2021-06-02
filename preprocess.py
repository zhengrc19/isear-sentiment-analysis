import pandas
import re
import os

def preprocess_df(s):
    s = re.sub("n't", ' not', s)
    s = re.sub(r"\\", '', s)
    return s

def preprocess(data_dir='./data'):
    for phase in ['train', 'valid', 'test']:
        df = pandas.read_csv(os.path.join(data_dir, f'isear_{phase}.csv'))
        if phase == 'train': 
            df.drop(3510, inplace=True)
        elif phase == 'valid':
            df.drop(1222, inplace=True)
        else:
            df.drop(697, inplace=True)
        df['sentence'] = df['sentence'].apply(preprocess_df)
        for i, row in df.iterrows():
            if 'no response' in row['sentence'].lower():
                #print(i, row['label'], row['sentence'])
                df.drop([i], inplace=True)
        df.to_csv(os.path.join(data_dir, f'{phase}.csv'))

if __name__ == '__main__':
    preprocess()