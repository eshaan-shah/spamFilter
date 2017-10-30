'''
Author: Eshaan Shah
Spam Filter using Naive Bayesian Classification, 
Support Vector Machines and Decision Tree Learning.
Takes as input training and testing set directory. 
Computes the confusion matrix, and model prediction accuracy.
'''

import os
import numpy as np
import argparse
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def predictResults(testingFeatureMatrix, testingLabels, model):
    '''
    Predicts the labels for test emails
    Computes the confusion matrix using the predicted and known labels
    param testingFeatureMatrix: Feature Matrix of the testing set 
    param testingLabels: Labels of the testing set emails (spam or not spam)
    param model: The model to be used
    return: confusion matrix, model prediction accuracy 
    '''
    result = model.predict(testingFeatureMatrix)
    confusionMatrix = confusion_matrix(testingLabels, result)
    fpr, tpr, thresholds = metrics.roc_curve(testingLabels, result, pos_label=1)
    accuracy = metrics.auc(fpr, tpr)
    return confusionMatrix, accuracy * 100

def trainModel(trainingFeatureMatrix, trainingLabels, model):
    '''
    Trains the model based on the feature matrix and the labels
    param trainingFeatureMatrix: Feature Matrix of the training set 
    param trainingLabels: Labels of the training set emails (spam or not spam)
    param model: The model to be used
    return: trained model
    '''
    model.fit(trainingFeatureMatrix, trainingLabels)
    return model

def buildFeatureMatrix(dictionaryOfWords, path):
    '''
    Builds the feature matrix for training as well as testing set using the top 3000 words obtained
    The dimensions of feature matrix are: No. of emails x 3000
    It also generates a label matrix, containing the known email labels (spam, or not spam)
    param dictionaryOfWOrds: dictionary containing top 3000 words
    param path: path of the directory containing emails
    return: list of lists Feature Matrix, list Email Labels
    '''
    labelForEmails = list([])
    for dirPath, dirName, fileName in os.walk(path):
        for dName in dirName:
            emails = os.listdir(os.path.join(dirPath, dName))
            labelForEmails.append(len(emails))
    totalEmails = labelForEmails[0] + labelForEmails[1]
    featureMatrix = np.zeros((totalEmails, 3000))
    emailLabels = np.zeros(totalEmails)
    emailLabels[labelForEmails[0] : labelForEmails[0] + labelForEmails[1]] = 1
    emailNumber = 0
    for dirPath, dirName, fileName in os.walk(path):
        for dName in dirName:
            emails = os.listdir(os.path.join(dirPath, dName))
            for email in emails:
                file = os.path.join(os.path.join(dirPath, dName), email)
                try:
                    with open(file, encoding="latin-1") as openFile:
                        for i, line in enumerate(openFile):
                            words = line.strip().split()
                            for word in words:
                                wordID = 0
                                for i, d in enumerate(dictionaryOfWords):
                                    if d[0] == word:
                                        wordID = i
                                        featureMatrix[emailNumber, wordID] = words.count(word)
                    emailNumber += 1
                except:
                    pass
    return featureMatrix, emailLabels

def createWordDictionary(srcPath):
    '''
    Creates a word dictionary by sequentially processing emails from training set
    Uses this dictionary to obtain a dictionary of top 3000 words
    These top 3000 words will be used as features
    param srcPath: path of the training set directory
    return: Dictionary of top 3000 words using training set
    '''
    wordList = list([])
    for dirPath, dirName, fileName in os.walk(srcPath):
        for dName in dirName:
            emails = os.listdir(os.path.join(dirPath, dName))
            for email in emails:
                file = os.path.join(os.path.join(dirPath, dName), email)
                try:
                    with open(file, encoding='latin-1') as openEmail:
                        for i, line in enumerate(openEmail):
                            words = line.strip().split()
                            for word in words:
                                if len(word) <= 3 or word == "Subject:":
                                    pass
                                else:
                                    wordList.append(word)
                except:
                    pass
    dictionaryOfWords = Counter(wordList)
    dictionaryOfWords = dictionaryOfWords.most_common(3000)
    return dictionaryOfWords

def argumentParsing():
    '''
    Parses the command line arguments
    The arguments parsed are: training set path, testing set path
    Displays appropriate help messages
    return: String train path, String test path
    '''
    argParser = argparse.ArgumentParser(description="SpamFilter")
    argParser.add_argument("trainPath", type=str,
                           help="Training data path required")
    argParser.add_argument("testPath", type=str,
                           help="Testing data path required")
    args = argParser.parse_args()
    trainPath = args.trainPath
    testPath = args.testPath
    return trainPath, testPath

def main():
    '''
    Main method
    Calls other functions
    return: None
    '''
    trainPath, testPath = argumentParsing()

    print("")
    print("Start: Create Word Dictionary...")
    wordDictionary = createWordDictionary(trainPath)
    print("End: Create Word Dictionary...")
    print("")

    print("Start: Buliding Feature Matrix for Training Data...")
    trainingFeatureMatrix, trainingLabels = buildFeatureMatrix(wordDictionary, trainPath)
    print("End: Buliding Feature Matrix for Training Data...")
    print("")

    print("Start: Buliding Feature Matrix for Testing Data...")
    testingFeatureMatrix, testingLabels = buildFeatureMatrix(wordDictionary, testPath)
    print("End: Buliding Feature Matrix for Testing Data...")
    print("")

    print("Start: Training GaussianNB")
    gaussianNbModel = trainModel(trainingFeatureMatrix, trainingLabels, GaussianNB())
    print("End: Training GaussianNB")
    print("")

    print("Start: Training MultinomialNB")
    multinomialNbModel = trainModel(trainingFeatureMatrix, trainingLabels, MultinomialNB())
    print("End: Training MultinomialNB")
    print("")

    print("Start: Training LinearSVC")
    linearSvcModel = trainModel(trainingFeatureMatrix, trainingLabels, LinearSVC())
    print("End: Training LinearSVC")
    print("")

    print("Start: Predicting using GaussianNB")
    confusionMatrix, acccuracy = predictResults(testingFeatureMatrix, testingLabels, gaussianNbModel)
    print(confusionMatrix)
    print(acccuracy)
    print("End: Predicting using GaussianNB")
    print("")

    print("Start: Predicting using MultinomialNB")
    confusionMatrix, acccuracy = predictResults(testingFeatureMatrix, testingLabels, multinomialNbModel)
    print(confusionMatrix)
    print(acccuracy)
    print("End: Predicting using MultinomialNB")
    print("")

    print("Start: Predicting using LinearSVC")
    confusionMatrix, acccuracy = predictResults(testingFeatureMatrix, testingLabels, linearSvcModel)
    print(confusionMatrix)
    print(acccuracy)
    print("End: Predicting using LinearSVC")
    print("")

if __name__ == '__main__':
    main()
