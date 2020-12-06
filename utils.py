import numpy as np 
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
from tensorflow.image import random_crop
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2

PATH ='00074.jpg'
DIRECTORY ='../datasets/near_duplicates/'
def prepare_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims =np.array([img_array]) 
    return preprocess_input(img_array_expanded_dims)

def prepare_image_query(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded_dims =np.array([img_array]) 
    return preprocess_input(img_array_expanded_dims)

def preprocess_crop(img_array):
    img_array_expanded_dims =np.array([img_array]) 
    return preprocess_input(img_array_expanded_dims)

def prepare_template_image(path):
    img = load_img(path, target_size=(112, 112))
    img_array = img_to_array(img)
    img_array = pad_image(img_array)
    img_array_expanded_dims =np.array([img_array]) 
    return preprocess_input(img_array_expanded_dims)

def pad_image(img, desiredX=224, desiredY=224):
    shape = img.shape
    xDiff = desiredX - shape[0]
    xLeft = xDiff//2
    xRight = xDiff-xLeft

    yDiff = desiredY - shape[1]
    yLeft = yDiff//2
    yRight = yDiff - yLeft

    return np.pad(img,((xLeft,xRight),(yLeft,yRight),(0,0)), mode='constant')

# Similarites
def similarities(kmeans,img_path,df,nb_similar=10):
    encoding = df.loc[img_path][:-1] # Feedforward NN 
    c = kmeans.predict(encoding.values.reshape(1,-1))    
    x = df[df['cluster']==c[0]].iloc[:,:-1]
    localSimilarity = {}
    l= list(x.index)
    l.remove(img_path)
    for img in l : 
        sim = 1- distance.cosine(x.loc[img], encoding)
        localSimilarity[img]= sim
        sort_localSim = sorted(localSimilarity.items(), key=lambda x: x[1], reverse=True)
    return sort_localSim[0:nb_similar]
def image_read(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
# Add class here
def generate_crop(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img = preprocess_input(img_array)
    # Modify this to be random 
    image = random_crop(img, size=[112, 112, 3])
    # Add Padding 
    cropped_padded = pad_image(image.numpy())
    return cropped_padded

def generate_duplicate_list(dir_path,n_random=10):
    crops={}
    for dir in os.listdir(dir_path): 
        dum =[]
        for i in range(n_random):
            dum.append(generate_crop(DIRECTORY+dir))
        crops[dir]=dum      
    return crops

def generate_data(cropped_dict,n_exemples=2):
    '''
    Params : 
    dict of image labels and ex
    ---
    Returns X a list of two [(N,224,224,3),(N,224,224,3)]
    
    and y a label 
    '''
    list_dict = list(enumerate(cropped_dict.keys()))
    N = 2 *n_exemples * len(list_dict)
    X =[np.empty((N,224,224,3)),np.empty((N,224,224,3))]
    y=np.zeros(N)
    index = 0 
    
    while index < N :
        for i in range(len(list_dict)):
            #prepare image
            query = prepare_image_query(DIRECTORY+list_dict[i][1])
            # Positive 
            for k in range(n_exemples):
                #Add positif exemple
                n = np.random.randint(0,9)
                crop = cropped_dict[list_dict[i][1]][n]
                #print(crop.shape)
                # templ = preprocess_crop(crop)
                label = 1
                X[0][index,],X[1][index,]=query,crop
                y[index]=label
                index = index+1                
           ## Negatives

            for k in range(n_exemples):
           #Add positif exemple
                n = np.random.randint(0,9)
                # Prendre l'exemple d'avant normalement pas ouf mais a refaire            
                crop = cropped_dict[list_dict[i-1][1]][n]      
                label = 0
                X[0][index,],X[1][index,]=query,crop
                y[index]=label
                index = index+1
        return X, y


def template_matching(query_image_path,template_image_path,model):
    query = prepare_image(query_image_path)
    template = prepare_template_image(template_image_path)
    pred = model.predict([query,template])
    
    fig, ax = plt.subplots(1,2,figsize=(18, 10))
    ax[0].imshow(query[0])
    ax[1].imshow(template[0])
    print('Prediction score :',pred)
    plt.savefig('result.png')