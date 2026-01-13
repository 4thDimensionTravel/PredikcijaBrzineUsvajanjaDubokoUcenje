import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
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


def resnet50(img_size=64, podaci_iz_tabele=12):

    #input_shape: Tvoja veličina slika
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    #ZAMRZAVANJE bitno
    base_model.trainable = False 
    
    input_img = Input(shape=(img_size, img_size, 3), name='slika_input')
    
    #Propuštamo sliku kroz ResNet
    x = base_model(input_img)

    
    #sabiramo sve bitne karakteristike u jedan vektor
    x = GlobalAveragePooling2D()(x) 
    
    x = Dense(128, activation='relu')(x) 
    x = Dropout(0.5)(x) # Dropout da sprečimo da se model previše osloni na slike 
    
    input_tab = Input(shape=(podaci_iz_tabele,), name='tabela_input')
    
    y = Dense(32, activation='relu')(input_tab)
    y = BatchNormalization()(y) # Stabilizuje učenje ## tabelarni podaci su na razlicitim skalama tj. vrednostima i onda mi normalizujemo 
    y = Dense(16, activation='relu')(y)
    
    combined = Concatenate()([x, y])
    
    z = Dense(64, activation='relu')(combined)
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

model2 = resnet50(img_size=64, podaci_iz_tabele=tabela_za_trening.shape[1])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model2.fit(
    x=[slike1, tabela_za_trening], 
    y=y,
    epochs=10, 
    batch_size=32,
    validation_split=0.2
)

model2.summary()