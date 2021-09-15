from tqdm import tqdm
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def make_test_data(train_data):
    
    VAL_PCT = 0.15  # lets reserve 10% of our data for validation
    val_size = int(len(train_data)*VAL_PCT)
    print(val_size)

    train_data = train_data[:-val_size]

    test_data = train_data[-val_size:]

    return test_data, train_data
        
def make_training_data():

    img_size = 64
    cats = 'convlution_neural_net/PetImages/Cat'
    dogs = 'convlution_neural_net/PetImages/Dog'
    Lables = {cats: 0, dogs: 1}

    training_data = []
    cat_count = 0
    dog_count = 0
    
    for lable in Lables:
        print(lable)
        for f in tqdm(os.listdir(lable)):
            try:
                path = os.path.join(lable, f)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                img = cv.resize(img, (img_size, img_size))
                training_data.append([np.array(img), np.eye(2)[Lables[lable]]])

                if lable == cats:
                    cat_count += 1
                elif lable == dogs:
                    dog_count += 1
            except Exception as e:
                #print(str(e))
                pass
    np.random.shuffle(training_data)
    print(f'Cats: {cat_count}, Dogs: {dog_count}')
    
    return training_data


x = make_training_data()
test_data, train_data = make_test_data(x)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

print(train_data[0])
print(train_data[1])
print(test_data[0])

np.save('convlution_neural_net/train_data.npy', train_data)
np.save('convlution_neural_net/test_data.npy', test_data)