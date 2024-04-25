# Initialize the dataset
from torch.utils.data import Dataset, Subset, DataLoader

class OrderBookDataset(Dataset):
    
    def __init__(self, order_book_df, num_events, predict_seconds):
        order_book_array = order_book_df.drop('Time_Delta', axis=1).values  # Convert DataFrame to numpy array
        X = []
        Y = []
        nanoseconds =  predict_seconds * 1e9
        for i in range(num_events, len(order_book_array)):
            X.append(order_book_array[i - num_events : i])
            sum_nano = 0
            for j in range(i, len(order_book_array)):
                sum_nano += order_book_df.loc[j, 'Time_Delta']
                if sum_nano >= nanoseconds:
                    if order_book_df.loc[j, 'MID_PRICE'] > order_book_df.loc[i, 'MID_PRICE']:
                        Y.append(2)
                    elif order_book_df.loc[j, 'MID_PRICE'] < order_book_df.loc[i, 'MID_PRICE']:
                        Y.append(1)
                    else:
                        Y.append(0)
                    break
        self.price_seq = X[:len(Y)]
        self.label = Y

    def __len__(self):
        return len(self.price_seq)
    
    def __getitem__(self, index):
        return (self.price_seq[index],  self.label[index])

def split_dataset(dataset, split_ratio):
    
    # Calculate the sizes of the splits
    train_size = int(split_ratio * len(dataset))
    remaining_ratio = (1 - split_ratio)/2
    test_size = int(remaining_ratio* len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    # Create indices for the splits
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):]
    return Subset(dataset, train_indices), Subset(dataset, val_indices ), Subset(dataset, test_indices)

def get_data_loaders(dataset, split_ratio, batch_size):
    train_set, val_set, test_set = split_dataset(dataset, split_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader