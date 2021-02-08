import numpy as np
import os
import glob
import math
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from sklearn.utils import shuffle
import sklearn
import random

#Pasta de onde os arquivos são retirados para o processamento
source_path = '/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons-numpy2/'

#Pasta para onde os arquivos serão encaminhados após o processamento
destination_path = '/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/'

biggerFile = 'S006C001P023R001A014.skeleton.npy.csv'
#Tranforma os dados para deixá-los mais parecidos com os da MSRC-12
def transform_data(source_path, destination_path):
    os.chdir(source_path)
    for file in glob.glob('*.npy'):
        print(file) #Imprime na tela o arquivo que está sendo processado
        data = np.load(source_path+file,allow_pickle=True) #abre o arquivo no caminho especificado
        data_item = data.item(0) #tranforma o dicionário em item para poder acessar por posição 
        data_skeleton = data_item['skel_body0'] #acessa apenas os dados relacionados ao skeleton
        new_data = np.zeros(shape=(data_skeleton.shape[0],data_skeleton[0].shape[0],4)) #cria um vetor vazio para adicionar a separação das juntas
        #Percorre todo o data_skeleton
        for i in range(data_skeleton.shape[0]):
            for j in range(data_skeleton.shape[1]):
                for k in range(4):
                    if (k==3): 
                        new_data[i,j,k] = 1 #adiciona o 1 ao final de cada junta no novo array
                    else:
                        new_data[i,j,k] = data_skeleton[i,j,k] #salva os dados no novo array
      
        #Tranforma o tamanhho do array para (n_frames,n_juntas*4) x+y+z+separação = 4
        final_data = new_data.reshape(new_data.shape[0],new_data.shape[1]*new_data.shape[2])

        #Excluindo as juntas extras 20,21,22,23 e 24
        for junta in range(final_data.shape[1]-1,0,-1):
            if (junta>=80):
                final_data = np.delete(final_data,junta,1)
            else:
                pass
        
        #Salva os dados processados em um arquivo .csv
        np.savetxt(destination_path+file+'.csv', final_data, delimiter=' ')

        #Imprime na tela o progresso em relação a quantidade de arquivos na pasta
        progress=+1
        print(str(int(100 * progress/len(glob.glob("*.npy")))) + "%")

        print(final_data.shape)
        
    #return final_data

#transform_data(source_path, destination_path)

#Procura o arquivo com maior número de frames para posteriormente fazer a deformação temporal
def bigger_file(source_path):
    bigger = -math.inf
    os.chdir(source_path)
    for file in glob.glob('*.csv'):
        print(file)
        data = pd.read_csv(source_path+file)
        if (data.shape[0]>bigger):
            bigger = data.shape[0]
            big_file = file
    print("---------------------------Maior arquivo---------------------------")
    print("Número de frames:" ,bigger, "Arquivo:", big_file)
    #return bigger, file

#bigger_file(destination_path)

def calculate_dtw(biggerFile):
    source_path = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/"
    os.chdir(source_path)
    file2 = biggerFile # maior tamanho
    progress = 0
    bigger_file = pd.read_csv(source_path+file2, delimiter=" ")
    X1 = bigger_file.to_numpy()
    for file in glob.glob("*.csv"):
        print(file)
        if(file.split(".")[0] != file2):
            files = pd.read_csv(source_path+file, delimiter=" ")
            X2 = files.to_numpy()
            distance, path = fastdtw(X1, X2, dist=euclidean)
            newX2 = np.zeros([len(path), 80])
            for i in range(len(path)):
                newX2[i, :] = X2[path[i][1], :] #x2 com tamanho do x1 a partir do fastdtw

            print (newX2.shape)
            # write new x2
            np.savetxt('data_dtw/' + file, newX2, delimiter=' ', fmt= '%f')
        progress += 1
        print(str(int(100 * progress/len(glob.glob("*.csv")))) + "%")

#calculate_dtw(biggerFile)

def select_files():
    source = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/12_gestures/"
    destination = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/12_gestures/camera_003/"
    os.chdir(source)
    for file in glob.glob("*.csv"):
        file_name = file.split(".")[0]
        gesture = int(file_name.split("A")[1])
        aux = file_name.split("C")[1]
        camera = aux.split("P")[0]
        if gesture<13:
            if camera=='003':
                os.link(source+file_name+'.csv',destination+file_name+'.csv')
                os.link(source+file_name+'.tagstream',destination+file_name+'.tagstream')

#select_files()

def select_24_files():
    source = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/"
    destination = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/24_gestures/"
    os.chdir(source)
    for file in glob.glob("*.csv"):
        file_name = file.split(".")[0]
        gesture = int(file_name.split("A")[1])
        if gesture<25:
            os.link(source+file_name+'.skeleton.npy.csv',destination+file_name+'.csv')
            os.link(source+file_name+'.tagstream',destination+file_name+'.tagstream')

    source = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/evaluate_Data/"
    destination = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/24_gestures/"
    os.chdir(source)
    for file in glob.glob("*.csv"):
        file_name = file.split(".")[0]
        gesture = int(file_name.split("A")[1])
        if gesture<25:
            os.link(source+file_name+'.skeleton.npy.csv',destination+file_name+'.csv')
            os.link(source+file_name+'.tagstream',destination+file_name+'.tagstream')

#select_24_files()
   

#cria os arquivos pickle para treinamento
def createPickleFiles():
    path = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/evaluate_Data/"
    os.chdir(path)
    evaluateData = np.zeros([len(glob.glob("*.csv")), 299,80])
    evaluateTag = np.zeros([len(glob.glob("*.csv"))])
    print("criando evaluatingData e evaluatingTag")
    for i, file in enumerate(glob.glob("*.csv")):
        file_name = file.split(".")[0]
        columns = np.arange(80)
        files = pd.read_csv(path+file_name+".skeleton.npy.csv", delimiter=" ",header=None, names=columns)
        tag_files = pd.read_csv(path+file_name+".tagstream", delimiter=" ", header=None, names='y')
        X = files.to_numpy()
        Y = tag_files.to_numpy()
        #(X, Y, tagset, n) = load_file(file.split(".")[0], pathSource)
        rX = np.reshape(X[:299, :80], [299, 80])
        rY = np.array([np.where(Y[:] == 1)[0]])
        evaluateData [i][:] = rX.astype(np.float32)
        evaluateTag [i] = rY.astype(np.int32)
        print(file_name)
        
    source_path = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/"
    os.chdir(source_path)
    trainingData = np.zeros([len(glob.glob("*.csv")), 299,80])
    trainingTag = np.zeros([len(glob.glob("*.csv"))])
    print("criando trainingData e trainingTag")
    for i, file in enumerate(glob.glob("*.csv")):
        file_name = file.split(".")[0]
        columns = np.arange(80)
        files = pd.read_csv(source_path+file_name+".skeleton.npy.csv", delimiter=" ",header=None, names=columns)
        tag_files = pd.read_csv(source_path+file_name+".tagstream", delimiter=" ", header=None, names='y')
        X = files.to_numpy()
        Y = tag_files.to_numpy()
        #(X, Y, tagset, n) = load_file(file.split(".")[0], pathSource)
        rX = np.reshape(X[:299, :80], [299, 80])
        rY = np.array([np.where(Y[:] == 1)[0]])
        trainingData [i][:] = rX.astype(np.float32)
        trainingTag [i] = rY.astype(np.int32)
        print(file_name)
        #print(str(int(100 * i/len(glob.glob("*.csv")))) + "%")

    print("escrevendo arquivos")
    #pathDestination = "/Users/julin/OneDrive/8SEMESTRE/Projeto_Integrador/Dataset/MicrosoftGestureDataset/MicrosoftGestureDataset-RC/pickle_data/"
    pathDestination = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/pickle_49_gestures-4files/"
    os.chdir(pathDestination)
    output_train_data = ('train_data')
    output_train_tag = ('train_tag')
    output_eval_data = ('eval_data')
    output_eval_tag = ('eval_tag')
    np.save(output_train_data, trainingData, allow_pickle=True, fix_imports=True)
    np.save(output_train_tag, trainingTag, allow_pickle=True, fix_imports=True)
    np.save(output_eval_data, evaluateData, allow_pickle=True, fix_imports=True)
    np.save(output_eval_tag, evaluateTag, allow_pickle=True, fix_imports=True)
    print("escrita dos arquivos concluida")

#createPickleFiles()

def create_labels_49():
    source_path = "/Users/julin/OneDrive/10SEMESTRE/TCC/nturgbd_skeletons_s001_to_s017/processed_data/data_dtw/"
    os.chdir(source_path)
    for file in glob.glob('*.csv'):
        file_name = file.split(".")[0]
        gesture = int(file_name.split("A")[1])
        y = np.zeros(49)
        y[gesture-1] = 1
        np.savetxt(file_name+'.tagstream', y, fmt='%d')
        print(file_name)

#create_labels_49()

