import torch
import argparse
from TextLSTM import TextLSTM
from TextCNN import TextCNN
from TextMLP import TextMLP
import preprocess
import dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Test script for ISEAR dataset. Written in 2021 by Ray Zheng as homework assignment for Intro to AI course.')
parser.add_argument('--model', type=str, default="cnn", help='Which model to test. Can choose between "cnn", "lstm", and "mlp".')
parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
parser.add_argument('--data_dir', type=str, default='./data', help="Directory for test set csv files.")
parser.add_argument('--emb_dim', type=int, default=300, help="Word embedding dimension.")

args = parser.parse_args()

if args.model not in ['cnn', 'lstm', 'mlp']:
    raise ValueError(f'Can only accept model "cnn", "lstm", or "mlp". Given model "{args.model}" not available.')

BATCH_SIZE = args.batch_size
EMBEDDINGS_DIMENSION = args.emb_dim # word embedding dimension
FIXED_SENTENCE_LEN = 60
NUM_SENTIMENT_CLASSES = 7

try:
    print("Creating dataset and iterator from csv files...")
    device, TEXT, train_iterator, valid_iterator, test_iterator = dataset.initialize_dataset(BATCH_SIZE, EMBEDDINGS_DIMENSION, args.data_dir)
except:
    print("Couldn't find preprocessed file. Preprocessing...")
    preprocess.preprocess(args.data_dir)
    print("Retrying create dataset and iterator from csv files...")
    device, TEXT, train_iterator, valid_iterator, test_iterator = dataset.initialize_dataset(BATCH_SIZE, EMBEDDINGS_DIMENSION, args.data_dir)


# general hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = EMBEDDINGS_DIMENSION
num_output_nodes = NUM_SENTIMENT_CLASSES
dropout = 0

# LSTM hyperparameters
hidden_dim = 64
num_layers = 2
bidirection = True

# CNN/MLP hyperparameters
num_features = 100
sentence_len = FIXED_SENTENCE_LEN
conv_windows = [3,4,5]

#instantiate the model
if args.model == 'cnn':
    Model = TextCNN(
        vocab_size = size_of_vocab, 
        emb_dim = embedding_dim, 
        feature_size = num_features, 
        num_classes = num_output_nodes, 
        sent_len = sentence_len, 
        conv_sizes = conv_windows,
        dropout = dropout
    )
elif args.model == 'lstm':
    Model = TextLSTM(
        vocab_size = size_of_vocab, 
        embedding_dim = embedding_dim, 
        hidden_dim = hidden_dim,
        output_dim = num_output_nodes, 
        n_layers = num_layers, 
        bidirectional = bidirection, 
        dropout = dropout
    )
else:
    Model = TextMLP(
        vocab_size = size_of_vocab,
        emb_dim = embedding_dim,
        num_classes = num_output_nodes,
        sent_len = sentence_len,
        dropout = dropout
    )
Model.to(device)

criterion = torch.nn.CrossEntropyLoss()

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    model.eval()
    all_predictions = []
    groud_truths = []
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating test set"):
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            all_predictions.extend(predictions.argmax(1))
            groud_truths.extend(batch.label)
            epoch_acc += (predictions.argmax(1) == batch.label).sum().item()
            epoch_count += batch.label.size(0)
            epoch_loss += loss.item()
    return epoch_loss / epoch_count, epoch_acc / epoch_count, all_predictions, groud_truths

ckpt = torch.load(f'{args.model}_saved_weights.pt', map_location=device)
Model.load_state_dict(ckpt)
print(f"Loaded {args.model} checkpoint.")

test_loss, test_acc, y_predicted, y_truth = evaluate(Model, test_iterator, criterion)

# print(y_predicted, y_truth)
print('Accuracy:', test_acc)
print('Macro F1:', f1_score(y_truth, y_predicted, average='macro'))
print('Micro F1:', f1_score(y_truth, y_predicted, average='micro'))