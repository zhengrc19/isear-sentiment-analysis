import torch
import argparse
from models.TextLSTM import TextLSTM
from models.TextCNN import TextCNN
from models.TextMLP import TextMLP
import preprocess
import dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Test script for ISEAR dataset. Written in 2021 by Ray Zheng as homework assignment for Intro to AI course.')
parser.add_argument('--model', type=str, default="cnn", help='Which model to test. Can choose between "cnn", "lstm", and "mlp".')
parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
parser.add_argument('--data_dir', type=str, default='./data', help="Directory for test set csv files.")
parser.add_argument('--emb_dim', type=int, default=300, help="Word embedding dimension.")
parser.add_argument('--lstm_hidden_dim', type=int,
                    default=64, help="Number of features in LSTM hidden state.")
parser.add_argument('--lstm_num_layers', type=int,
                    default=1, help="Number of recurrent layers of LSTM. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. ")
parser.add_argument('--lstm_bidirectional', type=bool,
                    default=True, help="If True, becomes a bidirectional LSTM. ")
parser.add_argument('--cnn_num_features', type=int,
                    default=200, help="Number of channels (features) produced by the convolution.")
parser.add_argument('--cnn_window_sizes', type=int, nargs='+',
                    default=[2, 3, 4, 5], help="Size of the convolving kernel, or in 1d terms window size.")

args = parser.parse_args()

if args.model not in ['cnn', 'lstm', 'mlp']:
    raise ValueError(f'Can only accept model "cnn", "lstm", or "mlp". Given model "{args.model}" not available.')

# general hyperparameters
BATCH_SIZE = args.batch_size
EMBEDDINGS_DIMENSION = args.emb_dim # word embedding dimension
FIXED_SENTENCE_LEN = 60
NUM_SENTIMENT_CLASSES = 7

# training hyperparameters
DROPOUT = 0

# LSTM hyperparameters
LSTM_HIDDEN_DIM = args.lstm_hidden_dim
LSTM_NUM_LAYERS = args.lstm_num_layers
LSTM_BIDIRECTIONAL = args.lstm_bidirectional

# CNN hyperparameters
CNN_NUM_FEATURES = args.cnn_num_features
CNN_WINDOW_SIZES = args.cnn_window_sizes

try:
    print("Creating dataset and iterator from csv files...")
    DEVICE, TEXT, train_iterator, valid_iterator, test_iterator = dataset.initialize_dataset(BATCH_SIZE, EMBEDDINGS_DIMENSION, args.data_dir)
except:
    print("Couldn't find preprocessed file. Preprocessing...")
    preprocess.preprocess(args.data_dir)
    print("Retrying create dataset and iterator from csv files...")
    DEVICE, TEXT, train_iterator, valid_iterator, test_iterator = dataset.initialize_dataset(BATCH_SIZE, EMBEDDINGS_DIMENSION, args.data_dir)


# general hyperparameters
size_of_vocab = len(TEXT.vocab)

#instantiate the model
if args.model == 'cnn':
    Model = TextCNN(
        vocab_size = size_of_vocab, 
        emb_dim = EMBEDDINGS_DIMENSION, 
        feature_size = CNN_NUM_FEATURES, 
        num_classes = NUM_SENTIMENT_CLASSES, 
        sent_len = FIXED_SENTENCE_LEN, 
        conv_sizes = CNN_WINDOW_SIZES,
        dropout = DROPOUT
    )
elif args.model == 'lstm':
    Model = TextLSTM(
        vocab_size = size_of_vocab, 
        embedding_dim = EMBEDDINGS_DIMENSION, 
        hidden_dim = LSTM_HIDDEN_DIM,
        output_dim = NUM_SENTIMENT_CLASSES, 
        n_layers = LSTM_NUM_LAYERS, 
        bidirectional = LSTM_BIDIRECTIONAL, 
        dropout = DROPOUT
    )
else:
    Model = TextMLP(
        vocab_size = size_of_vocab,
        emb_dim = EMBEDDINGS_DIMENSION,
        num_classes = NUM_SENTIMENT_CLASSES,
        sent_len = FIXED_SENTENCE_LEN,
        dropout = DROPOUT
    )
Model.to(DEVICE) 

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
            all_predictions.extend(predictions.argmax(1).cpu())
            groud_truths.extend(batch.label.cpu())
            epoch_acc += (predictions.argmax(1) == batch.label).sum().item()
            epoch_count += batch.label.size(0)
            epoch_loss += loss.item()
    return epoch_loss / epoch_count, epoch_acc / epoch_count, all_predictions, groud_truths

ckpt = torch.load(f'{args.model}_saved_weights.pt', map_location=DEVICE)
Model.load_state_dict(ckpt)
print(f"Loaded {args.model} checkpoint.")

test_loss, test_acc, y_predicted, y_truth = evaluate(Model, test_iterator, criterion)

#print(y_predicted)
#print(y_truth)
print('Test Set Accuracy:', test_acc)
print('Test Set Macro F-score:', f1_score(y_truth, y_predicted, average='macro'))
print('Test Set Micro F-score:', f1_score(y_truth, y_predicted, average='micro'))