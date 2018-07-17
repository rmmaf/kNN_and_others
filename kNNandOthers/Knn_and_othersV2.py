# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import matplotlib.pyplot as plt
import numpy as np



def loadDataset(filename1, filename2, quantity, trainingSet=[], testSet=[]):
    with open(filename1, 'rU') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(quantity):#deixar como float os valores do data set para poder calcula-los
                dataset[x][y] = float(dataset[x][y])
            trainingSet.append(dataset[x])
    with open(filename2, 'rU') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(quantity):  # deixar como float os valores do data set para poder calcula-los
                dataset[x][y] = float(dataset[x][y])
            testSet.append(dataset[x])

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

def newEtcNN(filename1, filename2, quantity, k):
    # prepare data
    trainingSet = []
    testSet = []
    loadDataset(filename1, filename2, quantity, trainingSet, testSet)
    print 'Training set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    #calculo KDN
    testSetKDN = []
    for x in range(len(testSet)):
        testSetKDN.append(getkDN(trainingSet, testSet[x], k))
    #media GeralKDN
    mediaGeralKdn = np.mean(testSetKDN)

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
    accuracyKNN = repr(accuracy)
    accuracyKNNNumber = accuracy

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
    accuracyANN = repr(accuracy)
    accuracyANNNumber = accuracy

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
    accuracyWNN = repr(accuracy)
    accuracyWNNNumber = accuracy
    print('Accuracy WNN: ' + repr(accuracy) + '%')

    fatorSoma = float(1.0/float(k))
    count = float(0.0)
    eixoYK = []
    eixoYW = []
    eixoYA = []
    eixoX = []
    #kdnMean = []
    while count < 1.0:
        acertosK = 0.0
        acertosW = 0.0
        acertosA = 0.0
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
                contagem = contagem + 1.0
        if contagem == 0.0:
            contagem = 1.0
        eixoX.append(float((count * 2.0 + fatorSoma) / 2.0))  # pegar o ponto medio do intervalo
        eixoYK.append(float(acertosK / contagem) * 100.0)
        eixoYW.append(float(acertosW / contagem) * 100.0)
        eixoYA.append(float(acertosA / contagem) * 100.0)
        count = count + fatorSoma
    count = 0.0
    kdnDeviation = []
    print mediaGeralKdn
    while count < 1.0:
        aux = []
        for x in range(len(testSetKDN)):
            if (count <= testSetKDN[x] < (count + fatorSoma)):
                aux.append(float((testSetKDN[x] - float(mediaGeralKdn)) ** 2.0))
        if len(aux) != 0:
            kdnDeviation.append(float((sum(aux) / len(aux)) ** 0.5))
        count = count + fatorSoma
    return eixoX, eixoYA, eixoYK, eixoYW, kdnDeviation

def listSum(list1, list2):
    if len(list1) == len(list2):
        listAux = []
        for x in range(len(list1)):
            listAux.append(list1[x] + list2[x])
        return listAux
    else:
        return 0

def toFloat(array):
    for x in range(len(array)):
        array[x] = float(array[x])
def main():
    eixoX = []
    eixoYA = []
    eixoYK = []
    eixoYW = []
    desvio = []
    g0 = operator.itemgetter(0)
    g1 = operator.itemgetter(1)
    g2 = operator.itemgetter(2)
    g3 = operator.itemgetter(3)
    g4 = operator.itemgetter(4)
    result = newEtcNN("cell0.csv", "cell9.csv", 2, 7)
    desvio = g4(result)
    if not eixoX:
        eixoX = g0(result)
        eixoYA = g1(result)
        eixoYK = g2(result)
        eixoYW = g3(result)
    else:
        eixoX = listSum(eixoX, g0(result))
        eixoYA = listSum(eixoYA, g1(result))
        eixoYK = listSum(eixoYK, g2(result))
        eixoYW = listSum(eixoYW, g3(result))
    toFloat(eixoYW)
    toFloat(eixoYA)
    toFloat(eixoYK)

    eixoYK.pop()
    eixoYA.pop()
    eixoYW.pop()
    eixoX.pop()
    print desvio
    print eixoYK
    print eixoYA
    print eixoYW
    print eixoX
    plt.errorbar(eixoX, eixoYA, desvio)
    plt.errorbar(eixoX, eixoYK, desvio)
    plt.errorbar(eixoX, eixoYW, desvio)

    plt.axis([0.00, 1.00, 0.00, 100.00])
    plt.show()

main()
