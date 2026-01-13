import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, AveragePooling2D 
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os


folder = 'RAF-PetFinder-dataset-main/Data/train_images/'

breeds = pd.read_csv('RAF-PetFinder-dataset-main/Data/breed_labels.csv')
#print(breeds.head)

states = pd.read_csv('RAF-PetFinder-dataset-main/Data/state_labels.csv')
#print(states.head)

test = pd.read_csv('RAF-PetFinder-dataset-main/Data/test/test.csv')
#print(test.head)

train = pd.read_csv('RAF-PetFinder-dataset-main/Data/train.csv')

color = pd.read_csv('RAF-PetFinder-dataset-main/Data/color_labels.csv')

def slike(folder, train):
    lista_slika = []
    validni_indeksi = [] 
    
    for i, petId in enumerate(train['PetID'][:2000]): 
       
        img_path = os.path.join(folder, f"{petId}-1.jpg")
            
        if os.path.exists(img_path):  
            try: 
                img = Image.open(img_path).convert('RGB')
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0
                
                lista_slika.append(img_array)
                validni_indeksi.append(i) 
            except:
                pass
        else:
            pass
            
    # Vracamo i slike i indekse
    return np.array(lista_slika), validni_indeksi

def lenet5(img_size=64, podaci_iz_tabele=10):
    input_img = Input(shape=(img_size, img_size, 3), name='slika_input')
    
    #Prvi blok: 6 filtera, kernel 5x5, Tanh aktivacija, Average Pooling
    x = Conv2D(6, (5, 5), activation='tanh')(input_img) 
    x = AveragePooling2D(pool_size=(2, 2))(x)

    #Drugi blok: 16 filtera, a gldea x(blok) 5x5x6
    x = Conv2D(16, (5, 5), activation='tanh')(x) 
    x = AveragePooling2D(pool_size=(2, 2))(x)
    
    #Peglanje
    x = Flatten()(x) 
    
    # Specifični gusti slojevi za LeNet (120 -> 84) ##120 i 84 origniali brojevi iz LeNet arh. odredili su ih 1998 zasto sto najbolje rade na
    x = Dense(120, activation='tanh')(x) 
    x = Dense(84, activation='tanh')(x)
    
    
    input_tab = Input(shape=(podaci_iz_tabele,), name='tabela_input')
    
    y = Dense(16, activation='relu')(input_tab) 
    
    y = Dense(8, activation='relu')(y)

    #spajanje
    combined = Concatenate()([x, y])
    
    z = Dense(32, activation='relu')(combined)
    z = Dropout(0.3)(z)
    
    output = Dense(5, activation='softmax', name='predikcija')(z) 

    model = Model(inputs=[input_img, input_tab], outputs=output)
    
    return model


slike1, indeksi = slike(folder, train)

kolone = ['Type', 'Age', 'Breed1', 'Gender', 'Color1', 
                      'MaturitySize', 'FurLength', 'Vaccinated', 
                      'Sterilized', 'Health', 'Fee', 'State']



podaci = train.iloc[:2000].iloc[indeksi]
tabela_za_trening = podaci[kolone].fillna(0).values 
y = podaci['AdoptionSpeed'].values 

model = lenet5(img_size=64, podaci_iz_tabele=tabela_za_trening.shape[1])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', #Koristimo ovo ako su targeti 0, 1, 2... 
    metrics=['accuracy']
)


if y is not None:
    print("Počinjem trening...")
    history = model.fit(
        x=[slike1, tabela_za_trening], #Dva ulaza u listi
        y=y,
        epochs=10, 
        batch_size=32, 
        validation_split=0.2 #20% podataka koristi za proveru (validaciju)
    )
else:
    print("Ne mogu da treniram bez AdoptionSpeed-a.")

model.summary()