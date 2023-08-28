from constant import START_TOKEN, END_TOKEN
from ds import *

def get_vocab():
    
    dataset = ds()
    x_train, y_train = dataset.get_train()
    
    # for i, text in enumerate(x_train):
    #     if not isinstance(text, str):
    #         print(f'Item {i} in x_train is not a string: {text}')
        
    # words = [START_TOKEN, END_TOKEN] + list(set(word for text in x_train for word in text.split()))
    words = [START_TOKEN, END_TOKEN] + list(set(word for text in x_train if isinstance(text, str) for word in text.split()))

    vocab = {
        word: idx for idx, word in enumerate(words)
    }
    # id_to_word = {
    #     v: k for k, v in word_to_id.items()
    # }

    return vocab



