import urllib.request
import pandas as pd

class ds:
    def __init__(self):
        file_path1 = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
        file_path2 = urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
        
        self.train_data = pd.read_table('ratings_train.txt')
        self.test_data = pd.read_table('ratings_test.txt')

        self.x_train = list(self.train_data['document'])[:5000]
        self.y_train = list(self.train_data['label'])[:5000]

        self.x_test = list(self.test_data['document'])[:1000]
        self.y_test = list(self.test_data['label'])[:1000]

    def get_train(self):
        return self.x_train, self.y_train


    def get_test(self):
        return self.x_test, self.y_test



    
    