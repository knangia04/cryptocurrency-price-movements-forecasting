import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import cohen_kappa_score, classification_report
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import training
import preprocess
import training_dataset
import model


parser = argparse.ArgumentParser(description='LOB RNN Model: Main Function')
parser.add_argument('--data_file', type=str, default='../data',
                    help='location of market data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of rnn (LSTM, GRU)')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='hidden_size in rnn model')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--predict_events', type=int, default=1,
                    help='how many events in the future to predict')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--sequence_length', type=int, default=60,
                    help='number of events in the sequence')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Laod and precoess data
df = pd.read_csv(args.data_file)
new_df = preprocess.process_data(df)
dataset = training_dataset.OrderBookDataset(new_df, args.sequence_length, args.predict_events)
train_loader, val_loader, test_loader = training_dataset.get_data_loaders(dataset, 0.8, args.batch_size)

model = model.LSTMModel(len(dataset[0][0][0]), args.hidden_size , args.num_layers).to(device)

# At any point you can hit Ctrl + C to break out of training early.
try:
    training.train(model, train_loader, val_loader, args.num_epochs, args.learning_rate, args.save)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open('rnn_model.pt', 'rb') as f:
    model = torch.load(f)

# Run on test data
acc,  precision, recall, f1score = training.evaluate(model, test_loader)
print('=' * 89)
print('---- End of training ----')
print(f"acc: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1score:.3f}")
print('=' * 89)

sns.set()
# Plot the loss of train and valid set after each epoch
plt.plot(Loss[1:])
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(('Train', 'Valid'))
plt.show()
