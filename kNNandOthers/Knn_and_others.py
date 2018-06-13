# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt



def loadDataset(filename, split, quantity, trainingSet=[], testSet=[]):
    with open(filename, 'rU') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(quantity):#deixar como float os valores do data set para poder calcula-los

                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
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


def newEtcNN(split, filename, quantity, k):
    # prepare data
    trainingSet = []
    testSet = []
    # split = 0.67
    loadDataset(filename, split, quantity, trainingSet, testSet)
    print 'Training set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    #calculo KDN
    testSetKDN = []
    for x in range (len(testSet)):
        testSetKDN.append(getkDN(trainingSet, testSet[x], k))
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

    fatorSoma = float(1.0/7.0)
    count = float(0.0)
    eixoYK = []
    eixoYW = []
    eixoYA = []
    eixoX = []
    while count < 1.0:
        acertosK = 0.0
        acertosW = 0.0
        acertosA = 0.0
        contagem = 0.0
        for x in range(len(testSetKDN)):
            if(count <= testSetKDN[x] < (count + fatorSoma)):
                if(acertosKnn[x]):
                    acertosK = acertosK + 1.0
                if(acertosWnn[x]):
                    acertosW = acertosW + 1.0
                if(acertosAnn[x]):
                    acertosA = acertosA + 1.0
                contagem = contagem + 1.0
        if contagem == 0.0:
            contagem = 1.0
        eixoX.append(float((count*2.0 + fatorSoma)/2.0))#pegar o ponto medio do intervalo
        eixoYK.append(float(acertosK/contagem) * 100.0)
        eixoYW.append(float(acertosW / contagem) * 100.0)
        eixoYA.append(float(acertosA / contagem) * 100.0)
        count = count + fatorSoma
    #plt.plot(eixoX, eixoYK, 'r--', eixoX, eixoYW, 'b--', eixoX, eixoYA, 'g--')

    plt.figure(1)  # the first figure
    plt.plot(eixoX, eixoYK, 'rs')
    plt.ylabel('Precisao Knn')
    plt.xlabel('KDN')

    plt.figure(2)  # the first figure
    plt.plot(eixoX, eixoYA, 'bs')
    plt.ylabel('Precisao Ann')
    plt.xlabel('KDN')

    plt.figure(3)  # the first figure
    plt.plot(eixoX, eixoYW, 'gs')
    plt.ylabel('Precisao Wnn')
    plt.xlabel('KDN')
    plt.show()

newEtcNN(0.8, 'whiteWine.csv', 11, 7)

def etcNN(split, filename, quantity, k):

    # prepare data
    trainingSet = []
    testSet = []
    #split = 0.67
    loadDataset(filename, split, quantity, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    print 'kNN'
    # generate predictions
    predictions = []

    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
       # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy kNN: ' + repr(accuracy) + '%')
    accuracyKNN = repr(accuracy)
    accuracyKNNNumber = accuracy

    print 'aNN'
    radiusOfTrainingSet = []
    for x in range(len(trainingSet)):
        radius = getSphereRadius(trainingSet, trainingSet[x], 0)
        radiusOfTrainingSet.append(radius)
    predictions = []

    for x in range(len(testSet)):
        neighbors = getNeighborsWithRadius(trainingSet, testSet[x], k, radiusOfTrainingSet)
        result = getResponse(neighbors)
        predictions.append(result)
       # print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy aNN: ' + repr(accuracy) + '%')
    accuracyANN = repr(accuracy)
    accuracyANNNumber = accuracy


    #w-NN
    print 'w-NN'
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponseWithWeight(neighbors, testSet[x])
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    accuracyWNN = repr(accuracy)
    accuracyWNNNumber = accuracy
    print 'kNN accuracy: ' + accuracyKNN + '%' + ' aNN accuracy: ' + accuracyANN + '%' + ' wNN accuracy: ' + accuracyWNN + '%'
    #kDN
    averagekDN = 0.0;
    for x in range(len(trainingSet)):
        averagekDN = averagekDN + getkDN(trainingSet, trainingSet[x], k)#usando o mesmo k do kNN e aNN
    finalKDN = repr(float(float(averagekDN)/float(len(trainingSet))))
    finalKDNNumber = float(float(averagekDN)/float(len(trainingSet)))
    print "Average kDN: " + finalKDN
    retorno = [accuracyKNNNumber, accuracyANNNumber, accuracyWNNNumber, finalKDNNumber]
    return retorno