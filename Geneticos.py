'''
Josue Fernandez Diaz - 2013033195
Inteligencia Artificial - José Carranza

'''
#import numpy para el manejo de array
import numpy as np
#import random para obtener valores aleatorios del W
import random as rand
#import para el mandejo de graficos
import matplotlib.pyplot as plt
#import para el manejo de los datos Iris
from sklearn.datasets import load_iris
#import os para el manejo de las carpetas de CIFAR
import os


########################### Iris Data ###########################
'''iris = load_iris()
data = iris.data
labels = iris.target
names = iris.target_names
clases = 3'''

########################### CIFAR Data ###########################
#Funcion obtenida del CIFAR-10
#File: Nombre del archivo a decifrar.
#Retorna: Un diccionario con data y labels
#Data: Una serie de 10000 arrays(imagenes) cada uno presenta una cantidad de 3072 colores.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
data = []
labels = []
clases = 4

def getData():
    for i in range(1,6):
        diccionario = unpickle(os.getcwd()+'\\data_batch_'+str(i))
        for j in range(len(diccionario[b'data'])):
            if(diccionario[b'labels'][j]<5):
                labels.append(diccionario[b'labels'][j]-1)
                data.append(diccionario[b'data'][j][:1024])

getData()



precision = 15

veces=15000
#Funcion Principal

def Main():
    steps = 1
    #Cantidad de datos a recuperar  por iteracion
    logHistory = []
    lenData = len(data)//steps
    ini=0
    hingeloss = []
    fin=steps
    w=np.random.uniform(0, 8, size=(clases,len(data[0])+1))
    for i in range(2500):
        x = getX(steps, ini,data) #optiene un dato
        mat = w@x #multiplicacion de ambas matrices
        tab = HingeLoss(mat,labels[ini:fin]) #hingeloss por cada clase
        sum = np.sum(np.max(tab,axis=1)) #suma de los valores del Hinge
        logHistory.append((w,sum))
        hingeloss.append(sum)

        w = mutarW(w)
        if fin_Mutacion(hingeloss,precision):
            return (logHistory,hingeloss) 
        ini+=steps
        fin+=steps
        if(fin>lenData):
            ini=0
            fin=steps

    return (logHistory,hingeloss)

def fin_Mutacion(sum,prec):
    if(len(sum)<15):
        return False
    sum = np.transpose(sum)
    for i in range(10):
        if(sum[i+1]-sum[i]>prec):
            return False
    return True
    
########################### Funcion Hinge Loss ###########################
'''
Funcion que obtiene el funcion lost de una matriz
Parametros:
matWX: Matriz original
lbl: Etiquetas de las clases originales de las matrices.
'''
def HingeLoss(matWX, lbl):
    mat = np.transpose(matWX)
    lenClass = len(mat)
    lenData = len(mat[0])
    final = np.zeros(lenClass,dtype=object)
    for i in range(lenClass):
        loss = np.zeros(lenData)
        for j in range(lenData):
            if(j!=lbl[i]):
                loss[j] = max(0,mat[i][j]-mat[i][lbl[i]]+1)
        final[i]=loss
    return final.tolist()
                
    
########################### Funcion Obtener X ###########################
#Obtiene una porcion de los datos
#
def getX(steps,cont, data):
    X = np.zeros(steps,dtype=object)
    for j in range(steps):
        X[j]=np.append(data[j+cont],[1])
    return np.transpose(X.tolist())
    

################################# Mutar W ################################
'''
Funcion
'''
def mutarW(W):

    #Experimento del 7 al
    W = np.transpose(W)
    for i in range(len(W)):
        W[i] = np.roll(W[i],i)

    #W[0] = np.roll(W[0],1)
   # W[3] = np.roll(W[3],3)
    W = np.transpose(W)

    return W
    
    '''
    
    #Experimento del 4 al 6
    
    res=len(W)//2
    ini=0
    fin=res-1
    while(ini<fin):
        temp = W[ini][0]
        W[ini][0]=W[fin][0]
        W[fin][0]=temp
        ini+=1
        fin-=1
    ini=res
    fin=len(W)-1
    while(ini<fin):
        temp = W[ini][0]
        W[ini][0]=W[fin][0]
        W[fin][0]=temp
        ini+=1
        fin-=1
    return W
    '''
    

    '''
    #Experimento 1 al 3

    w = np.transpose(W)
    w[500] = np.flip(w[500],0)
    return np.transpose(w)
    '''
    
def pintar(getImg):
    from PIL import Image
    data = np.zeros( (32,32,3), dtype=np.uint8)
    cont=0
    for i in range(0,32):
        for j in range(0,32):
            data[i][j] = [getImg[cont],getImg[cont],getImg[cont]]
            cont+=1
    img = Image.fromarray(np.asarray(data),'RGB')
    img.save('img.png')
    img.show()


(m,lista)=Main()

plt.plot(lista)
plt.title("Experimento 8") #titulo general
plt.xlabel("Datos")   # Establece el título del eje x
plt.ylabel("Hinge Loss")   # Establece el título del eje y
plt.show()

'''
k=np.random.uniform(0, 8, size=(3,len(data[0])+1))
print("normal")
print(k)
mutarW(k)
print("mutada")
print(k)

from sklearn.datasets import load_iris

>>> data = load_iris()
>>> data.target[[10, 25, 50]]
array([0, 0, 1])
>>> list(data.target_names)
['setosa', 'versicolor', 'virginica']
'''



