# Example of kNN implemented from Scratch in Python

import csv
import math
import operator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from numpy import array
from sklearn.neighbors import KDTree

def toFloat(array):
    for x in range(len(array)):
        array[x] = float(array[x])

def loadDataset(filename, quantity, dataArray=[]):# "transforma" o csv num array
    with open(filename, 'rU') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(quantity):#deixar como float os valores do data set para poder calcula-los
                dataset[x][y] = float(dataset[x][y])
                toFloat(dataset[x])
            dataArray.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += math.fabs(instance1[x] - instance2[x])
    return distance

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

#implementacao do a-NN
def getSphereRadius(trainingSet, trainingInstance, e):#classe para pegar o raio da instancia do conjunto de treino
    radius = 1
    distances = []
    length = len(trainingInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(trainingInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    classInstance = trainingInstance[len(trainingInstance) - 1] #pega a classe da instancia atual
    for x in range(len(distances)):
        f = operator.itemgetter(0)
        aux = f(distances[x])#consigo o "trainingSet[x]" do distance
        classTrain = aux[len(aux) - 1]#consigo a classe do traingingSet[x] relacionada a distance[x]
        if(classInstance !=  classTrain):
            f1 = operator.itemgetter(1)#consigo pegar o dist de distances[x]
            radius = f1(distances[x])
            break
    return radius - e

def getNeighborsWithRadius(trainingSet, testInstance, k, radius):#classe para pegar os vizinhos mais proximos baseando-se na nova medida
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        if(radius[x] == 0):
            radius[x] = 1
        dist = (euclideanDistance(testInstance, trainingSet[x], length) / radius[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#distance weighted k-NN

def getResponseWithWeight(neighbors, testInstance):
    classVotes = {}
    weights = []
    length = len(testInstance) - 1
    for x in range(len(neighbors)):
        dist = euclideanDistance(testInstance, neighbors[x], length)
        if pow(dist, 2.0) != 0.0:
            weights.append(float(1.0/pow(dist, 2.0)))
        else:
            weights.append(float(1.0))
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += weights[x]
        else:
            classVotes[response] = weights[x]
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
#kDN
def getkDN(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    classInstance = testInstance[len(testInstance) - 1]  # pega a classe da instancia atual
    count = 0
    for x in range(k):
        f = operator.itemgetter(0)
        aux = f(distances[x])
        classTrain = aux[len(aux) - 1]
        if (classInstance != classTrain):
            count = count + 1
    return float(float(count)/float(k))

def KNNr(tree, data, indice, r):
    dist, ind = tree.query([data[indice]], k=r)  # consulta a arvore, distancia e indice dos pontos mais proximos
    lista = []
    for x in range(len(ind[0])):
        if(x != 0):
            lista.append(ind[0][x])
    lista = array(lista)
    return lista

def naturalNeighbor(trainingSet):
    newTrainingSet = trainingSet
    r = 1
    flag = 0
    naN_Edge = set()
    tree = KDTree(newTrainingSet, leaf_size=2) #implementacao da arvore kd do trainingSet
    naNum = []
    for x in range(len(newTrainingSet)):#setando o NaNum para zero em todas as instancias do trainning set
        naNum.append(0)
    cnt = []
    while flag == 0:
        for x in range(len(newTrainingSet)):
            vizinhos = KNNr(tree, newTrainingSet, x, r + 1)
            for y in range(len(vizinhos)):
                otherKnn = KNNr(tree, newTrainingSet, vizinhos[y], r)
                auxSet = set(otherKnn)
                auxSet.add(x)
                if x in set(otherKnn) & auxSet not in naN_Edge:
                    naN_Edge.union(auxSet)
                    naNum[x] = naNum[x] + 1
                    naNum[vizinhos[y]] = naNum[vizinhos[y]] + 1
        cnt.append(naNum.count(0))
        rep = float(cnt.count(naNum.count(0)) - 1)
        if naNum.count(0) == 0:
            flag = 1
        r = r + 1
    naNE = r - 1
    return naNE

def newEtcNN(trainingSet, testSet, k):
    # prepare data
    print 'Training set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    #calculo KDN
    testSetKDN = []
    for x in range(len(testSet)):
        testSetKDN.append(getkDN(trainingSet, testSet[x], k))
    #media GeralKDN

    print 'kNN'
    # generate predictions
    predictions = []
    acertosKnn = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosKnn.append(True)
        else:
            acertosKnn.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy kNN: ' + repr(accuracy) + '%')

    print 'NaN'
    # generate predictions
    predictions = []
    acertosNaN = []
    naturalK = naturalNeighbor(trainingSet)
    print "Natural K: " + str(naturalK)
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], naturalK)
        result = getResponse(neighbors)
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosNaN.append(True)
        else:
            acertosNaN.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy NaN: ' + repr(accuracy) + '%')


    print 'aNN'
    radiusOfTrainingSet = []
    for x in range(len(trainingSet)):
        radius = getSphereRadius(trainingSet, trainingSet[x], 0.0)
        radiusOfTrainingSet.append(radius)
    predictions = []
    acertosAnn = []
    for x in range(len(testSet)):
        neighbors = getNeighborsWithRadius(trainingSet, testSet[x], k, radiusOfTrainingSet)
        result = getResponse(neighbors)
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosAnn.append(True)
        else:
            acertosAnn.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy aNN: ' + repr(accuracy) + '%')

    print 'NaNaNN'
    radiusOfTrainingSet = []
    for x in range(len(trainingSet)):
        radius = getSphereRadius(trainingSet, trainingSet[x], 0.0)
        radiusOfTrainingSet.append(radius)
    predictions = []
    acertosNaNAnn = []
    for x in range(len(testSet)):
        neighbors = getNeighborsWithRadius(trainingSet, testSet[x], naturalK, radiusOfTrainingSet)
        result = getResponse(neighbors)
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosNaNAnn.append(True)
        else:
            acertosNaNAnn.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy NaNaNN: ' + repr(accuracy) + '%')

    # w-NN
    print 'w-NN'
    predictions = []
    acertosWnn = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponseWithWeight(neighbors, testSet[x])
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosWnn.append(True)
        else:
            acertosWnn.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy WNN: ' + repr(accuracy) + '%')

    # w-NN
    print 'NaN w-NN'
    predictions = []
    acertosNaNWnn = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], naturalK)
        result = getResponseWithWeight(neighbors, testSet[x])
        predictions.append(result)
        if result == testSet[x][-1]:
            acertosNaNWnn.append(True)
        else:
            acertosNaNWnn.append(False)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy NaN WNN: ' + repr(accuracy) + '%')

    fatorSoma = float(1.0/float(k))
    count = float(0.0)
    eixoYK = []
    eixoYW = []
    eixoYA = []
    eixoYN = []
    eixoYNA = []
    eixoYNW = []
    eixoX = []
    #kdnMean = []
    while count < 1.0:
        acertosK = 0.0
        acertosW = 0.0
        acertosA = 0.0
        acertosN = 0.0
        acertosNA = 0.0
        acertosNW = 0.0
        #mediaKdn = []
        contagem = 0.0
        for x in range(len(testSetKDN)):
            if (count <= testSetKDN[x] < (count + fatorSoma)):
                if (acertosKnn[x]):
                    acertosK = acertosK + 1.0
                if (acertosWnn[x]):
                    acertosW = acertosW + 1.0
                if (acertosAnn[x]):
                    acertosA = acertosA + 1.0
                if (acertosNaN[x]):
                    acertosN = acertosN + 1.0
                if (acertosNaNAnn[x]):
                    acertosNA = acertosNA + 1.0
                if (acertosNaNWnn[x]):
                    acertosNW = acertosNW + 1.0
                contagem = contagem + 1.0
        if contagem == 0.0:
            contagem = 1.0
        eixoX.append(float((count * 2.0 + fatorSoma) / 2.0))  # pegar o ponto medio do intervalo
        eixoYK.append(float(acertosK / contagem) * 100.0)
        eixoYW.append(float(acertosW / contagem) * 100.0)
        eixoYA.append(float(acertosA / contagem) * 100.0)
        eixoYN.append(float(acertosN / contagem) * 100.0)
        eixoYNA.append(float(acertosNA / contagem) * 100.0)
        eixoYNW.append(float(acertosNW / contagem) * 100.0)
        count = count + fatorSoma
    return eixoX, eixoYA, eixoYK, eixoYW, eixoYN, eixoYNA, eixoYNW

def listSum(list1, list2):
    if len(list1) == len(list2):
        listAux = []
        for x in range(len(list1)):
            listAux.append(list1[x] + list2[x])
        return listAux
    else:
        return 0
def arraySum(array1, array2, arrayResult):
    for y in range(len(array1)):
        arrayResult[y] = (array1[y] + array2[y])

def divideList (list, factor):
    for x in range(len(list)):
        list[x] = list[x]/factor

def potList (list, factor):
    for x in range(len(list)):
        list[x] = list[x]**factor
def main(file, k, qtd , fold, exportName):
    eixoX = []
    eixoYA = []
    eixoYK = []
    eixoYW = []
    eixoYN = []
    eixoYNA = []
    eixoYNW = []
    desvioA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desvioK = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desvioW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desvioN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desvioNA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desvioNW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    arrayResultado = []
    dataArray = []

    loadDataset(file, qtd, dataArray) #colocar todos os dados num unico array
    dataArray = array(dataArray)
    # data sample
    # prepare cross validation
    kfold = KFold(fold, True, 1)
    # enumerate splits
    for train, test in kfold.split(dataArray):
        arrayResultado.append(newEtcNN(dataArray[train], dataArray[test], k))
    #inicio codigo para obter a media dos testes
    for x in range(len(arrayResultado) - 1):
        if not eixoX:# para a primeira busca
            eixoX = arrayResultado[0][0]
            eixoYA = arrayResultado[0][1]
            eixoYK = arrayResultado[0][2]
            eixoYW = arrayResultado[0][3]
            eixoYN = arrayResultado[0][4]
            eixoYNA = arrayResultado[0][5]
            eixoYNW = arrayResultado[0][6]

            arraySum(eixoYA, arrayResultado[x + 1][1], eixoYA)
            arraySum(eixoYK, arrayResultado[x + 1][2], eixoYK)
            arraySum(eixoYW, arrayResultado[x + 1][3], eixoYW)
            arraySum(eixoYN, arrayResultado[x + 1][4], eixoYN)
            arraySum(eixoYNA, arrayResultado[x + 1][5], eixoYNA)
            arraySum(eixoYNW, arrayResultado[x + 1][6], eixoYNW)

        else:
            arraySum(eixoYA, arrayResultado[x + 1][1], eixoYA)
            arraySum(eixoYK, arrayResultado[x + 1][2], eixoYK)
            arraySum(eixoYW, arrayResultado[x + 1][3], eixoYW)
            arraySum(eixoYN, arrayResultado[x + 1][4], eixoYN)
            arraySum(eixoYNA, arrayResultado[x + 1][5], eixoYNA)
            arraySum(eixoYNW, arrayResultado[x + 1][6], eixoYNW)

    divideList(eixoYA, float(fold))
    divideList(eixoYW, float(fold))
    divideList(eixoYK, float(fold))
    divideList(eixoYN, float(fold))
    divideList(eixoYNA, float(fold))
    divideList(eixoYNW, float(fold))
    #fim codigo da media
    eixoYA.pop()
    eixoYW.pop()
    eixoYK.pop()
    eixoYN.pop()
    eixoYNA.pop()
    eixoYNW.pop()
    eixoX.pop()


    #inicio codigo desvio padrao
    for x in range(len(eixoX)):
        for y in range(len(arrayResultado)):
            desvioA[x] = desvioA[x] + (arrayResultado[y][1][x] - eixoYA[x])**2.0
            desvioK[x] = desvioK[x] + (arrayResultado[y][2][x] - eixoYK[x])**2.0
            desvioW[x] = desvioW[x] + (arrayResultado[y][3][x] - eixoYW[x]) ** 2.0
            desvioN[x] = desvioN[x] + (arrayResultado[y][4][x] - eixoYN[x]) ** 2.0
            desvioNA[x] = desvioNA[x] + (arrayResultado[y][5][x] - eixoYNA[x]) ** 2.0
            desvioNW[x] = desvioNW[x] + (arrayResultado[y][6][x] - eixoYNW[x]) ** 2.0
    divideList(desvioA, float(fold))
    divideList(desvioW, float(fold))
    divideList(desvioK, float(fold))
    divideList(desvioN, float(fold))
    divideList(desvioNA, float(fold))
    divideList(desvioNW, float(fold))
    potList(desvioA, 0.5)
    potList(desvioW, 0.5)
    potList(desvioK, 0.5)
    potList(desvioN, 0.5)
    potList(desvioNA, 0.5)
    potList(desvioNW, 0.5)
    #fim codigo desvio padrao
    print "eixoYAnn: " + str(eixoYA)
    print 'eixoYWnn: ' + str(eixoYW)
    print "eixoYKnn: " + str(eixoYK)
    print "eixoYNaNKnn: " + str(eixoYN)
    print "eixoYNaNAnn: " + str(eixoYNA)
    print "eixoYNaNWnn: " + str(eixoYNW)
    print "eixoX: " + str(eixoX)
    print 'desvioAnn: ' + str(desvioA)
    print 'desvioWnn: ' + str(desvioW)
    print 'desvioKnn: ' +  str(desvioK)
    print "desvioNaNKnn: " + str(desvioN)
    print  "desvioNaNAnn: " + str(desvioNA)
    print  "desvioNaNWnn: " + str(desvioNW)


    plt.errorbar(eixoX, eixoYA, desvioA)
    plt.errorbar(eixoX, eixoYK, desvioK)
    plt.errorbar(eixoX, eixoYW, desvioW)
    plt.errorbar(eixoX, eixoYN, desvioN)
    plt.errorbar(eixoX, eixoYNA, desvioNA)
    plt.errorbar(eixoX, eixoYNW, desvioNW)

    plt.axis([0.00, 1.00, 0.00, 100.00])
    plt.savefig(exportName, dpi=600)
    plt.show()

main("sonar.csv", 7, 60, 10, "sonar.pdf")
