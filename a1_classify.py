from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import numpy as np
import argparse
import sys
import os

file_temp = open("temp_result.txt", 'w')

def print_and_write(output):
    global file_temp
    file_temp.write(str(output) + "\n")
    print(output)
    return

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct = 0
    total = 0
    for i in range(4):
        correct = correct + C[i][i]
    for i in range(4):
        for j in range(4):
            total = total + C[i][j]
    if(total == 0):
        total = 1
    return(float(correct)/float(total))


def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall = np.zeros(4)
    total = 0
    for i in range(4):
        recall[i] = C[i][i]
        for j in range(4):
            total = total + C[i][j]
        if(total != 0):
            recall[i] = float(recall[i])/float(total)
        total = 0
    return(recall)

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    prec = np.zeros(4)
    total = 0
    for i in range(4):
        prec[i] = C[i][i]
        for j in range(4):
            total = total + C[j][i]
        if(total != 0):
            prec[i] = float(prec[i])/float(total)
        total = 0
    return(prec)

   

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')


    # Classifiers SVC



    data_array = np.load(filename)

    print("Shape of data array: " + str(data_array.shape))


    print(data_array[0].astype(int))
    print(data_array[1].astype(int))
    print(data_array[2].astype(int))
    print(data_array[12000].astype(int))
    print(data_array[22000].astype(int))
    print(data_array[32000].astype(int))

    #X = data_array[:, 0:173]
    #y = data_array[:, 173]
    
    X1 = data_array[0:100, 0:173]
    y1 = data_array[0:100, 173]

    X2 = data_array[10000:10100, 0:173]
    y2 = data_array[10000:10100, 173]

    X3 = data_array[20000:20100, 0:173]
    y3 = data_array[20000:20100, 173]

    X4 = data_array[30000:30100, 0:173]
    y4 = data_array[30000:30100, 173]
    ''' 
    X1 = data_array[0:10, 0:173]
    y1 = data_array[0:10, 173]

    X2 = data_array[10000:10010, 0:173]
    y2 = data_array[10000:10010, 173]

    X3 = data_array[20000:20010, 0:173]
    y3 = data_array[20000:20010, 173]

    X4 = data_array[30000:30010, 0:173]
    y4 = data_array[30000:30010, 173]
    '''


    X = np.concatenate((X1, X2, X3, X4))
    y = np.concatenate((y1, y2, y3, y4))

    print("Shape of X: " + str(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


    # A) LINEAR

    print_and_write("\n\nLinear SVC")

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []

    print_and_write("Linear test data...")
    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion_1 = confusion_matrix(true_class, classifications)
    print_and_write("SVC Linear confusion")
    print_and_write(confusion_1)

    accuracy_1 = accuracy(confusion_1)
    print_and_write("SVC Linear accuracy")
    print_and_write(accuracy_1)

    precision_1 = precision(confusion_1)
    print_and_write("SVC Linear precision")
    print_and_write(precision_1)

    recall_1 = recall(confusion_1)
    print_and_write("SVC Linear recall")
    print_and_write(recall_1)

    # B) RADIAL

    print_and_write("\n\nRadial SVC")

    clf = SVC(kernel='rbf', gamma=2)
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []


    print_and_write("Radial test data...")
    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion_2 = confusion_matrix(true_class, classifications)
    print_and_write("SVC Radial confusion")
    print_and_write(confusion_2)

    accuracy_2 = accuracy(confusion_2)
    print_and_write("SVC Radial accuracy")
    print_and_write(accuracy_2)

    precision_2 = precision(confusion_2)
    print_and_write("SVC Radial precision")
    print_and_write(precision_2)

    recall_2 = recall(confusion_2)
    print_and_write("SVC Radial recall")
    print_and_write(recall_2)


    # C) RandomForestClassifier

    print_and_write("\n\nRandomForestClassifier")

    clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []



    print_and_write("RFC test data...")
    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion_3 = confusion_matrix(true_class, classifications)
    print_and_write("RFC confusion")
    print_and_write(confusion_3)

    accuracy_3 = accuracy(confusion_3)
    print_and_write("RFC accuracy")
    print_and_write(accuracy_3)

    precision_3 = precision(confusion_3)
    print_and_write("RFC precision")
    print_and_write(precision_3)

    recall_3 = recall(confusion_3)
    print_and_write("RFC recall")
    print_and_write(recall_3)

    # D) MLPClassifier

    print_and_write("\n\nMLPClassifier")

    clf = MLPClassifier(alpha=0.05)
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []

    print_and_write("MLP test data...")
    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion_4 = confusion_matrix(true_class, classifications)
    print_and_write("MLP confusion")
    print_and_write(confusion_4)

    accuracy_4 = accuracy(confusion_4)
    print_and_write("MLP accuracy")
    print_and_write(accuracy_4)

    precision_4 = precision(confusion_4)
    print_and_write("MLP precision")
    print_and_write(precision_4)

    recall_4 = recall(confusion_4)
    print_and_write("MLP recall")
    print_and_write(recall_4)



    # E) AdaBoostClassifier

    print_and_write("\n\nAdaBoostClassifier")

    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []


    print_and_write("ABC test data...")
    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion_5 = confusion_matrix(true_class, classifications)
    print_and_write("ABC confusion")
    print_and_write(confusion_5)

    accuracy_5 = accuracy(confusion_5)
    print_and_write("ABC accuracy")
    print_and_write(accuracy_5)

    precision_5 = precision(confusion_5)
    print_and_write("ABC precision")
    print_and_write(precision_5)

    recall_5 = recall(confusion_5)
    print_and_write("ABC recall")
    print_and_write(recall_5)





    iBest = ''

    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()



    # TODO : complete each classification experiment, in sequence.
    
    class31(args.input)
    









