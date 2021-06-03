import torch
import argparse
from models.TextLSTM import TextLSTM
from models.TextCNN import TextCNN
from models.TextMLP import TextMLP
import preprocess
import dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train script for ISEAR dataset. Written in 2021 by Ray Zheng as homework assignment for Intro to AI course.')
parser.add_argument('--model', type=str, 
                    default="cnn", help='Which model to train. Can choose between "cnn", "lstm", and "mlp".')
parser.add_argument('--epoch', type=int, 
                    default=60, help="Number of epochs in training.")
parser.add_argument('--batch_size', type=int, 
                    default=64, help="Batch size.")
parser.add_argument('--data_dir', type=str, 
                    default='./data', help="Directory for train, validation, and test set csv files.")
parser.add_argument('--emb_dim', type=int, 
                    default=300, help="Word embedding dimension.")
parser.add_argument('--lr', type=float, 
                    default=0.5, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, 
                    default=0.01, help="Weight decay (L2 regularization) constant.")
parser.add_argument('--gamma', type=float, 
                    default=0.1, help="Multiplicative factor of learning rate decay.")
parser.add_argument('--step_size', type=int, 
                    default=15, help="Period of learning rate decay.")
parser.add_argument('--dropout', type=float, 
                    default=0.5, help="During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.")
parser.add_argument('--lstm_hidden_dim', type=int,
                    default=64, help="Number of features in LSTM hidden state.")
parser.add_argument('--lstm_num_layers', type=int,
                    default=1, help="Number of recurrent layers of LSTM. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. ")
parser.add_argument('--lstm_bidirectional', type=bool,
                    default=True, help="If True, becomes a bidirectional LSTM. ")
parser.add_argument('--cnn_num_features', type=int,
                    default=200, help="Number of channels (features) produced by the convolution.")
parser.add_argument('--cnn_window_sizes', type=int, nargs='+',
                    default=[2, 3, 4, 5], help="Size of the convolving kernel, or in 1d terms window size. To use multiple window sizes, run like so: python train.py --model cnn --cnn_window_sizes 2 3 4 5")

args = parser.parse_args()

if args.model not in ['cnn', 'lstm', 'mlp']:
    raise ValueError(f'Can only accept model "cnn", "lstm", or "mlp". Given model "{args.model}" not available.')

# general hyperparameters
N_EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
EMBEDDINGS_DIMENSION = args.emb_dim # word embedding dimension
FIXED_SENTENCE_LEN = 60
NUM_SENTIMENT_CLASSES = 7

# training hyperparameters
LR = args.lr
DROPOUT = args.dropout
WEIGHT_DECAY = args.weight_decay
GAMMA = args.gamma
LR_STEP_SIZE = args.step_size

# LSTM hyperparameters
LSTM_HIDDEN_DIM = args.lstm_hidden_dim
LSTM_NUM_LAYERS = args.lstm_num_layers
LSTM_BIDIRECTIONAL = args.lstm_bidirectional

# CNN hyperparameters
CNN_NUM_FEATURES = args.cnn_num_features
CNN_WINDOW_SIZES = args.cnn_window_sizes

def train(model, iterator, optimizer, criterion, epoch_num):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    model.train()  
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.label)
        #update loss and accuracy 
        epoch_acc += (predictions.argmax(1) == batch.label).sum().item()
        epoch_count += batch.label.size(0)
        epoch_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return epoch_loss / epoch_count, epoch_acc / epoch_count

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            epoch_acc += (predictions.argmax(1) == batch.label).sum().item()
            epoch_count += batch.label.size(0)
            epoch_loss += loss.item()
    return epoch_loss / epoch_count, epoch_acc / epoch_count

print(f"Preprocessing csv files in ./data ...")
preprocess.preprocess(args.data_dir)
print("Creating dataset and iterator from csv files...")
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

# freeze GloVe
pretrained_embeddings = TEXT.vocab.vectors
Model.embedding.weight.data.copy_(pretrained_embeddings)
for p in Model.embedding.parameters():
    p.requires_grad = False

#No. of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The {args.model.upper()} model has {count_parameters(Model):,} trainable parameters')

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_valid_loss = float('inf')
best_valid_acc = 0
best_epoch = 0

# Training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, LR_STEP_SIZE, gamma=GAMMA)

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(Model, train_iterator, optimizer, criterion, epoch + 1)
    valid_loss, valid_acc = evaluate(Model, valid_iterator, criterion)
    scheduler.step()
    train_losses.append(train_loss)
    train_accs.append(train_acc * 100)
    val_losses.append(valid_loss)
    val_accs.append(valid_acc * 100)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_valid_acc = valid_acc
        best_epoch = epoch + 1
        torch.save(Model.state_dict(), f'{args.model}_saved_weights.pt')
    print(f'Epoch {epoch+1}/{N_EPOCHS}:')
    print(f'\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.5f} |  Val. Acc: {valid_acc*100:.2f}%')

print(f"Training finished! Achieved lowest validation loss of {best_valid_loss:.7f} with accurary of {best_valid_acc*100:.2f}% at epoch {best_epoch}.")

plt.figure()
plt.title(f"{args.model.upper()} Loss Curve")
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(0, N_EPOCHS+1, 2))
plt.grid()
plt.legend()

plt.figure()
plt.title(f"{args.model.upper()} Acc Curve")
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(0, N_EPOCHS+1, 2))
plt.grid()
plt.legend()
plt.show()