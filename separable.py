import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, AveragePooling2D , SeparableConv2D, BatchNormalization, GlobalAveragePooling2D
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

def separable(img_size=64, podaci_iz_tabele=12):
    input_img = Input(shape=(img_size, img_size, 3), name='slika_input')
    
    x = Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(input_img) 
    x = BatchNormalization()(x) 

    # SeparableConv2D  == Conv2D 
    x = SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x) 

    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x) 
    
    x = SeparableConv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = SeparableConv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.4)(x)
    
    input_tab = Input(shape=(podaci_iz_tabele,), name='tabela_input')
    y = Dense(16, activation='relu')(input_tab)
    
    
    combined = Concatenate()([x, y])
    
    
    z = Dense(32, activation='relu')(combined)
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

model3 = separable(img_size=64, podaci_iz_tabele=tabela_za_trening.shape[1])

model3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


if len(slike1) > 0:
    history3 = model3.fit(
        x=[slike1, tabela_za_trening], 
        y=y,
        epochs=10, 
        batch_size=32,
        validation_split=0.2
    )
    
    print("\nBroj parametara Modela 3:")
    model3.summary()