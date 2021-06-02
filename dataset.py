import torch
#handling text data
from torchtext.legacy import data

def initialize_dataset(b_size = 64, emb_dim = 300, data_dir='./data'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emo2vec = {
        'joy'    : 0,
        'shame'  : 1,
        'fear'   : 2,
        'anger'  : 3,
        'sadness': 4,
        'guilt'  : 5,
        'disgust': 6
    }
    with open('./data/stopwords.txt') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.extend(
            ['á', "'s", ' ', '.', '(', ')', '-', '"', ',', '[', ']']
        )
        stopwords.remove('not')
        stopwords.remove('no')

    TEXT = data.Field(
        sequential=True, 
        tokenize='basic_english',
        batch_first=True, 
        include_lengths=True, 
        lower=True, 
        use_vocab=True,
        stop_words = stopwords
    )
    LABEL = data.LabelField(
        sequential=False, 
        use_vocab=False, 
        dtype = torch.long, 
        preprocessing=lambda x: emo2vec[x], 
        batch_first=True
    )

    #loading custom dataset
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path  = './data',
        train = 'train.csv', 
        validation = 'valid.csv',
        test = 'test.csv',
        format = 'csv',
        fields = [(None, None), (None, None), ('label', LABEL), ('text',TEXT)], 
        skip_header = True
    )

    TEXT.build_vocab(train_data, vectors=f'glove.42B.{emb_dim}d',min_freq=3)
    # print(TEXT.vocab.freqs.most_common(50))
    print(f'Constructed vocabulary of {len(TEXT.vocab)} words.')

    #Load an iterator
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = b_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch=True,
        device = device)
    
    return device, TEXT, train_iterator, valid_iterator, test_iterator

if __name__ == '__main__':
    initialize_dataset()