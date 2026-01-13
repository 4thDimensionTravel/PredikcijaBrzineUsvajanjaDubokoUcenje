import json
import pandas as pd
import sample_subset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os


breeds = pd.read_csv('RAF-PetFinder-dataset-main/Data/breed_labels.csv')
#print(breeds.head)

states = pd.read_csv('RAF-PetFinder-dataset-main/Data/state_labels.csv')
#print(states.head)

test = pd.read_csv('RAF-PetFinder-dataset-main/Data/test/test.csv')
#print(test.head)

color = pd.read_csv('RAF-PetFinder-dataset-main/Data/color_labels.csv')

def RasaBrojDrzava(test, states, breeds):
    test = test.merge(breeds[['BreedID', 'BreedName']], left_on='Breed1', right_on='BreedID', how='left')
    test.rename(columns={'BreedName': 'Breed1Name'}, inplace=True)
    test.drop('BreedID', axis=1, inplace=True)

    test = test.merge(breeds[['BreedID', 'BreedName']], left_on='Breed2', right_on='BreedID', how='left')
    test.rename(columns={'BreedName': 'Breed2Name'}, inplace=True)
    test.drop('BreedID', axis=1, inplace=True)

    test['BreedSum'] = test.apply(lambda row: 'Mixed Breed' if row['Breed2'] != 0 else row['Breed1Name'], axis=1)

    test = test.merge(states, left_on='State', right_on='StateID', how='left')

    dogs = test[test['Type'] == 1]
    cats = test[test['Type'] == 2]

    breed_state_counts_dog = dogs.groupby(['StateName', 'BreedSum']).size().reset_index(name='Count')
    breed_state_counts_dog = breed_state_counts_dog[breed_state_counts_dog['Count'] > 5]

    breed_state_counts_cat = cats.groupby(['StateName', 'BreedSum']).size().reset_index(name='Count')
    breed_state_counts_cat = breed_state_counts_cat[breed_state_counts_cat['Count'] > 5]

    top_breeds_per_state_dog = breed_state_counts_dog.sort_values(['StateName','Count'], ascending=[True, False]).groupby('StateName').head(5)
    top_breeds_per_state_cat = breed_state_counts_cat.sort_values(['StateName','Count'], ascending=[True, False]).groupby('StateName').head(5)

    plt.figure(figsize=(12,8))
    sns.barplot(data=top_breeds_per_state_dog, x='BreedSum', y='Count', hue='StateName')
    plt.title('Top 5 najzastupljenijih rasa pasa po državi')
    plt.xlabel('Rasa')
    plt.ylabel('Broj pasa')
    plt.legend(title='Država')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))
    sns.barplot(data=top_breeds_per_state_cat, x='BreedSum', y='Count', hue='StateName')
    plt.title('Top 5 najzastupljenijih rasa mačaka po državi')
    plt.xlabel('Rasa')
    plt.ylabel('Broj mačaka')
    plt.legend(title='Država')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
def BojaBroj(test, color):
    test = test.merge(color[['ColorID', 'ColorName']], left_on='Color1', right_on='ColorID', how='left')
    test.rename(columns={'ColorName': 'Color1Name'}, inplace=True)
    test.drop('ColorID', axis=1, inplace=True)

    test = test.merge(color[['ColorID', 'ColorName']], left_on='Color2', right_on='ColorID', how='left')
    test.rename(columns={'ColorName': 'Color2Name'}, inplace=True)
    test.drop('ColorID', axis=1, inplace=True)

    test = test.merge(color[['ColorID', 'ColorName']], left_on='Color3', right_on='ColorID', how='left')
    test.rename(columns={'ColorName': 'Color3Name'}, inplace=True)
    test.drop('ColorID', axis=1, inplace=True)

 
    test['ColorMix'] = test.apply(lambda row: 'Color Mixed' if (row['Color2'] != 0 or row['Color3'] != 0) else row['Color1Name'],axis=1)

    color_counts = test['ColorMix'].value_counts().reset_index()
    color_counts.columns = ['Color', 'Count']
    
    color = ['#4682B4', 'Brown', 'Black', '#F2F0EF', '#EAA221', '#FDFBD4', '#898989', '#FFDE21']

    plt.figure(figsize=(10, 6))
    sns.barplot(data=color_counts, x='Color', y='Count', palette=color)
    plt.title('Broj ljubimaca po boji')
    plt.xlabel('Boja')
    plt.ylabel('Broj ljubimaca')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    
def ImePol(test):
    
    test = test[test['Gender'].isin([1, 2])]

    name_gender_counts = test.groupby(['Gender', 'Name']).size().reset_index(name='Count')
    
    top_names_per_gender = (name_gender_counts.sort_values(['Gender', 'Count'], ascending=[True, False]).groupby('Gender').head(5))
    top_names_per_gender['Gender'] = top_names_per_gender['Gender'].map({1: 'Muški', 2: 'Ženski'})
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=top_names_per_gender, x='Name', y='Count', hue='Gender')
    plt.title('Top 5 najčešćih imena po polu')
    plt.xlabel('Ime ljubimca')
    plt.ylabel('Broj ljubimaca')
    plt.legend(title='Pol')
    plt.tight_layout()
    plt.show()

def PosetaVeterinaru(test):
    test['VisitedVet'] = test.apply(
        lambda row: 0 if (row['Vaccinated'] == 1 and row['Dewormed'] == 1 and 
                          row['Sterilized'] == 1 and row['Health'] == 1)
        else 1, axis=1
    )
    
    #test['HealthGroup'] = test['Health'].apply(lambda x: 2 if x == 2 else 3)

    vet_health_counts = test.groupby(['VisitedVet']).size().reset_index(name='Count')
    color = ['#4682B4', '#FFDE21']

    plt.figure(figsize=(8,6))
    sns.barplot(data=vet_health_counts, x='VisitedVet', y='Count', palette= color)
    plt.title('Broj zivotinja za usvajanje koji ne mora kod veterinara')
    plt.xlabel('Poseta veterinaru')
    plt.ylabel('Broj ljubimaca')
    plt.legend(title='Zdravstveno stanje \nPlava = ne mora da poseti\nZuta = mora da poseti')
    plt.tight_layout()
    plt.show()
    
def Starost(test):
    
    test['AgeNormalno'] = test['Age'].apply(lambda x: x/12)
    
    prosek_godina = test['AgeNormalno'].mean()
    
    plt.figure(figsize=(8,6))
    sns.histplot(data=test, x = 'AgeNormalno', bins=20, kde=True, color='skyblue')
    
    plt.axvline(prosek_godina, linestyle='--', linewidth=2)
    
    plt.title('Distribucija starosti ljubimaca')
    plt.xlabel('Starost (godine)')
    plt.ylabel('Broj ljubimaca')
    plt.tight_layout()
    plt.show()

'''
def Pokusaj(test):
    
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'
    lista = []
    
    for petId in test['PetID'][:10]:
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_array = np.array(img)

                wisina, sirina, kanal = img_array.shape
                sirina_start, sirina_end = sirina//4, 3*sirina//4
                ##sliku na samo centrali deo
                centar = img_array[:, sirina_start:sirina_end, :]

                # Pronalazak dominantne boje u centralnom delu
                # Flatten za sve piksele
                pixels = centar.reshape(-1, 3)
                # Brojanje pojavljivanja svake boje zaokruzeno
                pixels_zaokruzeno = (pixels // 10) * 10  # smanjuje broj unikatnih boja
                unique, counts = np.unique(pixels_zaokruzeno, axis=0, return_counts=True)
                dominant_color = unique[np.argmax(counts)]
                
                ime = test.loc[test['PetID'] == petId, 'Name'].iloc[0]
                
                ljubimac = {
                "PetID": petId,
                "Name": ime,
                "Slika": x,
                "DominantnaBoja": tuple(dominant_color)
            }
                lista.append(ljubimac)
                
                print(f"Pet name: {ime}, PetID:{petId}, Slika {x}, Dominantna boja (RGB): {dominant_color}")

            else:
                print(f"Slika {petId}-{x}.jpg jbg ne postoji")
'''

def BojaSlika(test):
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'

    # Uzimamo prvih 6 ljubimaca za pregled
    pet_ids = test['PetID'][:3]

    fig, axes = plt.subplots(len(pet_ids), 1, figsize=(10, 4 * len(pet_ids)))

    # Ako ima samo 1 red, napravi da axes bude lista
    if len(pet_ids) == 1:
        axes = [axes]

    for i, petId in enumerate(pet_ids):
        # Tražimo slike tipa petId-1.jpg, petId-2.jpg
        found = False
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            if os.path.exists(img_path):
                found = True
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # histograma za R, G, B
                colors = ('r', 'g', 'b')
                for j, col in enumerate(colors):
                    hist, bins = np.histogram(img_array[:,:,j].flatten(), bins=256, range=(0,256))
                    axes[i].plot(hist, color=col)
                
                #axes[i].set_title(f"PetID: {petId} - Histogram boja slike {x}")
                axes[i].set_xlabel("Intenzitet boje PetID: {petId} - Histogram boja slike {x}")
                axes[i].set_ylabel("Broj piksela")
                break

        if not found:
            axes[i].set_title(f"Slike za PetID {petId} nisu pronađene")
            #axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    
    
def OdnosStranica(test):
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'

    odnosi = []
    imena = []

    for petId in test['PetID'][:10]:
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                sirina, visina = img.size

                odnos = sirina / visina  # > 1.0 lendskejp, < 1.0 portret
                odnosi.append(odnos)

                ime = test.loc[test['PetID'] == petId, 'Name'].iloc[0]
                imena.append(f"{ime}-{x}")

    plt.figure(figsize=(10,4))
    plt.bar(imena, odnosi, color='orange')
    plt.xticks(rotation=45, ha='right')
    plt.title("Odnos stranica (širina / visina) po slici")
    plt.ylabel("Odnos stranica")
    plt.tight_layout()
    plt.show()
    
def ProsecnaSvetlina(test):
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'
    svetlina = []

    for petId in test['PetID'][:20]:
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')  # grayscale
                avg_brightness = np.mean(np.array(img))
                svetlina.append(avg_brightness)
                break  # samo prva slika po ljubimcu
    
    plt.hist(svetlina, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribucija prosečne svetline slika")
    plt.xlabel("Prosečna svetlina (0 = tamno, 255 = svetlo)")
    plt.ylabel("Frekvencija")
    plt.show()


def Kontrast(test):
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'

    kontrasti = []
    imena = []

    for petId in test['PetID'][:10]:
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L")  # grayscale
                img_array = np.array(img)

                std_val = np.std(img_array)

                ime = test.loc[test['PetID'] == petId, 'Name'].iloc[0] 
                imena.append(f"{ime}-{x}")
                kontrasti.append(std_val)

    plt.figure(figsize=(10,4))
    plt.bar(imena, kontrasti, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.title("Kontrast po slici")
    plt.ylabel("Standardna devijacija (kontrast)")
    plt.tight_layout()
    plt.show()


def EntropijaSlike(test):
    #Računa Šenonovu entropiju  za slike.
    folder = 'RAF-PetFinder-dataset-main/Data/test_images/'

    entropije = []
    imena = []

    for petId in test['PetID'][:10]:
        for x in range(1, 3):
            img_path = os.path.join(folder, f"{petId}-{x}.jpg")
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L") 
                img_array = np.array(img)

                # Izračunaj histogram bins=256 jer imamo vrednosti od 0 do 255
                histogram, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))

                #normalizacija
                prob_dist = histogram / histogram.sum()

                prob_dist = prob_dist[prob_dist > 0]

                entropy = -np.sum(prob_dist * np.log2(prob_dist))     #-sum(p * log2(p))

                ime = test.loc[test['PetID'] == petId, 'Name'].iloc[0]
                imena.append(f"{ime}-{x}")
                entropije.append(entropy)

    plt.figure(figsize=(10, 4))
    plt.bar(imena, entropije, color='#9b59b6') 
    plt.xticks(rotation=45, ha='right')
    plt.title("Entropija slike (Kompleksnost)")
    plt.ylabel("Entropija (biti)")
    
    plt.axhline(np.mean(entropije), color='red', linestyle='--', label='Prosečna entropija')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


    
#RasaBrojDrzava(test, states, breeds)
#BojaBroj(test, color)
#ImePol(test)
#PosetaVeterinaru(test)
#Starost(test)
#BojaSlika(test)
#ProsecnaSvetlina(test)
#Kontrast(test)
#OdnosStranica(test)
#EntropijaSlike(test)
