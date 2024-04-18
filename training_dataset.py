# Initialize the dataset
from torch.utils.data import Dataset, Subset

class OrderBookDataset(Dataset):
    
    def __init__(self, order_book_df, num_events, predict_seconds):
        order_book_array = order_book_df.drop('Time_Delta', axis=1).values  # Convert DataFrame to numpy array
        X = []
        Y = []
        nanoseconds =  predict_seconds * 1e9
        for i in range(num_events, len(order_book_array)): #Starting from num_events because we need num_events of previous data to predict the next one
            X.append(order_book_array[i - num_events : i, 0]) #Get the previous num_events data
            """ target_time = order_book_df.loc[i, "COLLECTION_TIME"]  + nanoseconds
            time_difference = abs(order_book_df['COLLECTION_TIME'] - target_time)
            # Find the index of the row with the smallest time difference
            closest_index = time_difference.idxmin()
            if order_book_df.loc[closest_index, 'MID_PRICE'] > order_book_df.loc[closest_index, 'MID_PRICE']: #If the price goes up
                Y.append(2) # 2 is for up
            elif order_book_df.loc[closest_index, 'MID_PRICE'] < order_book_df.loc[closest_index, 'MID_PRICE']: #``If the price goes down
                Y.append(1) # 1 is for down 
            else:
                Y.append(0)# 0 is for same"""
            print(f"index: {i}")
            sum_nano = 0
            for j in range(i, len(order_book_array)): 
                sum_nano += order_book_df.loc[j, 'Time_Delta']
                if sum_nano >= nanoseconds: #If the sum of nanoseconds is greater than the nanoseconds we want to predict
                    if order_book_df.loc[j, 'MID_PRICE'] > order_book_df.loc[i, 'MID_PRICE']: #If the price goes up
                        Y.append(2) # 2 is for up
                    elif order_book_df.loc[j, 'MID_PRICE'] < order_book_df.loc[i, 'MID_PRICE']: #``If the price goes down
                        Y.append(1) # 1 is for down 
                    else:
                        Y.append(0)# 0 is for same
                    break
        self.price_seq = X
        self.label = Y

    def __len__(self):
        return len(self.price_seq)
    
    def __getitem__(self, index):
        return (self.price_seq[index],  self.label[index])
