import pandas as pd

class DataHandler:

    def __init__(self):
        # read data
        (train_negative, train_positive, test_negative, test_positive) = self.read_data()

        # merging train and test data
        self.data = pd.concat([train_negative, train_positive, test_negative, test_positive], axis=0)
        self.rename_dataset_columns(self.data)

        # merging datasets
        self.train_data = pd.concat([train_positive, train_negative], axis=0)
        self.test_data = pd.concat([test_positive, test_negative], axis=0)

        # rename columns
        self.rename_dataset_columns(self.train_data)
        self.rename_dataset_columns(self.test_data)

        # drop duplicates
        self.train_data = self.train_data.drop_duplicates()
        self.test_data = self.test_data.drop_duplicates()

        # shuffle dataset
        self.train_data = self.shuffle_dataset(self.train_data)
        self.test_data = self.shuffle_dataset(self.test_data)

    def shuffle_dataset(self, dataset):
        return dataset.sample(frac=1)

    def rename_dataset_columns(self, dataset):
        dataset.rename(columns={0: 'label', 1: 'tweet'}, inplace=True )

    def read_data(self):
        path = "datasets/"
        train_negative = pd.read_csv(
            path + 'train_Arabic_tweets_negative_20190413.tsv', sep='\t',
            header=None)
        train_positive = pd.read_csv(
            path + 'train_Arabic_tweets_positive_20190413.tsv', sep='\t',
            header=None)
        test_negative = pd.read_csv(
            path + 'test_Arabic_tweets_negative_20190413.tsv', sep='\t',
            header=None)
        test_positive = pd.read_csv(
            path + 'test_Arabic_tweets_positive_20190413.tsv', sep='\t',
            header=None)

        return (train_negative, train_positive, test_negative, test_positive)

    def get_data(self):
        return (self.train_data, self.test_data, self.data)

