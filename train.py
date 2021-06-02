import torch
import argparse
from TextLSTM import TextLSTM
from TextCNN import TextCNN
from TextMLP import TextMLP
import preprocess
import dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Train script for ISEAR dataset. Written in 2021 by Ray Zheng as homework assignment for Intro to AI course.')
parser.add_argument('--model', type=str, default="cnn", help='Which model to train. Can choose between "cnn", "lstm", and "mlp".')
parser.add_argument('--epoch', type=int, default=20, help="Number of epochs in training.")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
parser.add_argument('--data_dir', type=str, default='./data', help="Directory for train, validation, and test set csv files.")
parser.add_argument('--emb_dim', type=int, default=300, help="Word embedding dimension.")
parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay (regularization) constant.")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout probability")

args = parser.parse_args()

if args.model not in ['cnn', 'lstm', 'mlp']:
    raise ValueError(f'Can only accept model "cnn", "lstm", or "mlp". Given model "{args.model}" not available.')

N_EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
EMBEDDINGS_DIMENSION = args.emb_dim # word embedding dimension
FIXED_SENTENCE_LEN = 60
NUM_SENTIMENT_CLASSES = 7

def train(model, iterator, optimizer, criterion, epoch_num):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    model.train()  
    for batch in tqdm(iterator, desc=f"Epoch {epoch_num}/{N_EPOCHS}"):
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.label)
        #update loss and accuracy 
        epoch_acc += (predictions.argmax(1) == batch.label).sum().item()
        epoch_count += batch.label.size(0)
        epoch_loss += loss.item()
        loss.backward()
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

print(f"Preprocessing csv files in {args.data_dir}...")
preprocess.preprocess(args.data_dir)
print("Creating dataset and iterator from csv files...")
device, TEXT, train_iterator, valid_iterator, test_iterator = dataset.initialize_dataset(BATCH_SIZE, EMBEDDINGS_DIMENSION, args.data_dir)

# general hyperparameters
size_of_vocab = len(TEXT.vocab)
embedding_dim = EMBEDDINGS_DIMENSION
num_output_nodes = NUM_SENTIMENT_CLASSES
dropout = args.dropout

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

# freeze GloVe
pretrained_embeddings = TEXT.vocab.vectors
Model.embedding.weight.data.copy_(pretrained_embeddings)
for p in Model.embedding.parameters():
    p.requires_grad = False

#No. of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The {args.model.upper()} model has {count_parameters(Model):,} trainable parameters')


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), weight_decay=args.weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_valid_loss = float('inf')
best_valid_acc = 0
best_epoch = 0

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(Model, train_iterator, optimizer, criterion, epoch + 1)
    valid_loss, valid_acc = evaluate(Model, valid_iterator, criterion)
    
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
    #print(f'Epoch {epoch+1}/{N_EPOCHS}:')
    print(f'\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.5f} |  Val. Acc: {valid_acc*100:.2f}%')

print(f"Training finished! Achieved lowest validation loss of {best_valid_loss:.7f} with accurary of {best_valid_acc*100:.2f}% at epoch {epoch}.")

plt.figure()
plt.title(f"{args.model.upper()} Loss Curve")
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(N_EPOCHS))
plt.grid()
plt.legend()

plt.figure()
plt.title(f"{args.model.upper()} Acc Curve")
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(N_EPOCHS))
plt.grid()
plt.legend()
plt.show()