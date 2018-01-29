from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
import numpy as np
import argparse
import sys
import os


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
parser.add_argument("--csv_out", default="garb.txt")
parser.add_argument("--result_out", default="garb.txt")
parser.add_argument("--data_size", type=int, default=10000)
parser.add_argument("--csv_3_1", default="a1_3.1.csv")
parser.add_argument("--csv_3_2", default="a1_3.2.csv")
parser.add_argument("--csv_3_3", default="a1_3.3.csv")
parser.add_argument("--part4_out", default="part4_out.txt")
parser.add_argument("--part4_csv", default="a1_3.4.csv")
args = parser.parse_args()


file_temp = open(args.result_out, 'w')
csv_temp = open(args.csv_out, 'w')
csv_3_1 = open(args.csv_3_1, 'w')
csv_3_2 = open(args.csv_3_2, 'w')
csv_3_3 = open(args.csv_3_3, 'w')
part_4_file = open(args.part4_out,'w')
part_4_csv = open(args.part4_csv, 'w')

def print_and_write(output):
    global file_temp
    file_temp.write(str(output) + "\n")
    print(output)
    return

def write_to_csv_4( classifier_type, accuracy, recall, precision, confusion ):
    part_4_csv.write(str(classifier_type) + ",")
    part_4_csv.write(str(accuracy) + ",")
    for element in recall:
        part_4_csv.write(str(element) + ",")
    for element in precision:
        part_4_csv.write(str(element) + ",")
    for i in range(4):
        for j in range(4):
            part_4_csv.write(str(confusion[i][j]))
            if(i != 3 or j != 3):
                part_4_csv.write(",")
    part_4_csv.write("\n")
   
def write_to_csv( classifier_type, accuracy, recall, precision, confusion ):
    csv_3_1.write(str(classifier_type) + ",")
    csv_3_1.write(str(accuracy) + ",")
    for element in recall:
        csv_3_1.write(str(element) + ",")
    for element in precision:
        csv_3_1.write(str(element) + ",")
    for i in range(4):
        for j in range(4):
            csv_3_1.write(str(confusion[i][j]))
            if(i != 3 or j != 3):
                csv_3_1.write(",")
    csv_3_1.write("\n")
   
def write_to_csv_part2(accuracy):
    csv_3_2.write(str(accuracy) + ",") 

def print_and_write4(output):
    part_4_file.write(str(output) + "\n")
    print(output)
    return

def write_to_csv_part4(output):
    part_4_csv.write(str(output) + "\n")
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


    #print(data_array[0].astype(int))
    #print(data_array[1].astype(int))
    #print(data_array[2].astype(int))
    #print(data_array[12000].astype(int))
    #print(data_array[22000].astype(int))
    #print(data_array[32000].astype(int))
    
    X = data_array[:, 0:173]
    y = data_array[:, 173]
    
    '''    
    X1 = data_array[0:100, 0:173]
    y1 = data_array[0:100, 173]

    X2 = data_array[10000:10100, 0:173]
    y2 = data_array[10000:10100, 173]

    X3 = data_array[20000:20100, 0:173]
    y3 = data_array[20000:20100, 173]

    X4 = data_array[30000:30100, 0:173]
    y4 = data_array[30000:30100, 173]
    '''
    '''
    X1 = data_array[0:args.data_size, 0:173]
    y1 = data_array[0:args.data_size, 173]

    X2 = data_array[10000:10000+args.data_size, 0:173]
    y2 = data_array[10000:10000+args.data_size, 173]

    X3 = data_array[20000:20000+args.data_size, 0:173]
    y3 = data_array[20000:20000+args.data_size, 173]

    X4 = data_array[30000:30000+args.data_size, 0:173]
    y4 = data_array[30000:30000+args.data_size, 173]
    
    best_acc = 0
    best = 0
    '''
    '''
    X = np.concatenate((X1, X2, X3, X4))
    y = np.concatenate((y1, y2, y3, y4))
    '''
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
    best = 1
    best_acc = accuracy_1

    precision_1 = precision(confusion_1)
    print_and_write("SVC Linear precision")
    print_and_write(precision_1)

    recall_1 = recall(confusion_1)
    print_and_write("SVC Linear recall")
    print_and_write(recall_1)

    write_to_csv(1, accuracy_1, recall_1, precision_1, confusion_1)

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

    if(accuracy_2 > best_acc):
        best_acc = accuracy_2
        best = 2

    precision_2 = precision(confusion_2)
    print_and_write("SVC Radial precision")
    print_and_write(precision_2)

    recall_2 = recall(confusion_2)
    print_and_write("SVC Radial recall")
    print_and_write(recall_2)


    write_to_csv(2, accuracy_2, recall_2, precision_2, confusion_2)

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

    if(accuracy_3 > best_acc):
        best_acc = accuracy_3
        best = 3

    precision_3 = precision(confusion_3)
    print_and_write("RFC precision")
    print_and_write(precision_3)

    recall_3 = recall(confusion_3)
    print_and_write("RFC recall")
    print_and_write(recall_3)


    write_to_csv(3, accuracy_3, recall_3, precision_3, confusion_3)


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

    if(accuracy_4 > best_acc):
        best_acc = accuracy_4
        best = 4

    precision_4 = precision(confusion_4)
    print_and_write("MLP precision")
    print_and_write(precision_4)

    recall_4 = recall(confusion_4)
    print_and_write("MLP recall")
    print_and_write(recall_4)


    write_to_csv(4, accuracy_4, recall_4, precision_4, confusion_4)


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

    if(accuracy_5 > best_acc):
        best_acc = accuracy_5
        best = 5

    precision_5 = precision(confusion_5)
    print_and_write("ABC precision")
    print_and_write(precision_5)

    recall_5 = recall(confusion_5)
    print_and_write("ABC recall")
    print_and_write(recall_5)


    write_to_csv(5, accuracy_5, recall_5, precision_5, confusion_5)



    iBest = best

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

    clf = 0

    if(iBest == 1):
        clf = SVC(kernel='linear')
    elif(iBest == 2):
        clf = SVC(kernel='rbf', gamma=2)
    elif(iBest == 3):
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif(iBest == 4):
        clf = MLPClassifier(alpha=0.05)
    elif(iBest == 5):
        clf = AdaBoostClassifier()

    



    clf.fit(X_train, y_train)

    classifications = []
    true_class = []



    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion = confusion_matrix(true_class, classifications)
 
    write_to_csv_part2(accuracy(confusion)) 

    X_1k = X_train[0:1000]
    y_1k = y_train[0:1000]   

    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, iBest, X_train_1k, X_test_1k, y_train_1k, y_test_1k):
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

    clf = None 

    print(iBest)

    # 1 K
    if(iBest == 1):
        clf = SVC(kernel='linear')
    elif(iBest == 2):
        clf = SVC(kernel='rbf', gamma=2)
    elif(iBest == 3):
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif(iBest == 4):
        clf = MLPClassifier(alpha=0.05)
    elif(iBest == 5):
        clf = AdaBoostClassifier()
    clf.fit(X_train_1k, y_train_1k)


    classifications = []
    true_class = []

    for i, x_test in enumerate(X_test_1k):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test_1k[i])

    confusion = confusion_matrix(true_class, classifications)
 
    csv_3_3.write("\n 1k_accuracy with top 5: " + str(accuracy(confusion)))

    # 32 K
    if(iBest == 1):
        clf = SVC(kernel='linear')
    elif(iBest == 2):
        clf = SVC(kernel='rbf', gamma=2)
    elif(iBest == 3):
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif(iBest == 4):
        clf = MLPClassifier(alpha=0.05)
    elif(iBest == 5):
        clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)


    classifications = []
    true_class = []

    for i, x_test in enumerate(X_test):
        pred_class = clf.predict(x_test.reshape(1, -1))
        classifications.append(pred_class)
        true_class.append(y_test[i])

    confusion = confusion_matrix(true_class, classifications)
 
    csv_3_3.write("\n 32k_accuracy with top 5: " + str(accuracy(confusion)))







def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')




    data_array = np.load(filename)

    print("Shape of data array: " + str(data_array.shape))


    #print(data_array[0].astype(int))
    #print(data_array[1].astype(int))
    #print(data_array[2].astype(int))
    #print(data_array[12000].astype(int))
    #print(data_array[22000].astype(int))
    #print(data_array[32000].astype(int))

    X = data_array[:, 0:173]
    y = data_array[:, 173]

    #X = X[0:100]
    #y = y[0:100]
    '''       
    X1 = data_array[0:50, 0:173]
    y1 = data_array[0:50, 173]

    X2 = data_array[10000:10050, 0:173]
    y2 = data_array[10000:10050, 173]

    X3 = data_array[20000:20050, 0:173]
    y3 = data_array[20000:20050, 173]

    X4 = data_array[30000:30050, 0:173]
    y4 = data_array[30000:30050, 173]
    '''
    '''
    X1 = data_array[0:args.data_size, 0:173]
    y1 = data_array[0:args.data_size, 173]

    X2 = data_array[10000:10000+args.data_size, 0:173]
    y2 = data_array[10000:10000+args.data_size, 173]

    X3 = data_array[20000:20000+args.data_size, 0:173]
    y3 = data_array[20000:20000+args.data_size, 173]

    X4 = data_array[30000:30000+args.data_size, 0:173]
    y4 = data_array[30000:30000+args.data_size, 173]
    
    best_acc = 0
    best = 0
    '''
    #X = np.concatenate((X1, X2, X3, X4))
    #y = np.concatenate((y1, y2, y3, y4))
    
    print("Shape of X: " + str(X.shape))

    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=42)

    j = 0

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        j = j + 1
        print("TRAIN:", train_index, "TEST:", test_index)
   
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #X_train = X_train[0:4000]
        #y_train = y_train[0:4000]

        data_size=25

        print(X_train.shape)

        X1 = X_train[0:data_size, 0:173]
        y1 = y_train[0:data_size]

        X2 = X_train[8000:8000+data_size, 0:173]
        y2 = y_train[8000:8000+data_size]

        X3 = X_train[16000:16000+data_size, 0:173]
        y3 = y_train[16000:16000+data_size]

        X4 = X_train[24000:24000+data_size, 0:173]
        y4 = y_train[24000:24000+data_size]
        

        X_train = np.concatenate((X1, X2, X3, X4))
        y_train = np.concatenate((y1, y2, y3, y4))



        data_size = 200

        X1 = X_test[0:data_size, 0:173]
        y1 = y_test[0:data_size]

        X2 = X_test[2000:2000+data_size, 0:173]
        y2 = y_test[2000:2000+data_size]

        X3 = X_test[4000:4000+data_size, 0:173]
        y3 = y_test[4000:4000+data_size]

        X4 = X_test[6000:6000+data_size, 0:173]
        y4 = y_test[6000:6000+data_size]
        

        X_test = np.concatenate((X1, X2, X3, X4))
        y_test = np.concatenate((y1, y2, y3, y4))
 



 
        write_to_csv_part4("\n\n FOLD: " + str(j))
        print_and_write4("\n\n FOLD: " + str(j))

        best = 0
        best_acc = 0

        # A) LINEAR

        print_and_write4("\n\nLinear SVC")
        
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)


        classifications = []
        true_class = []

        print_and_write4("Linear test data...")
        for i, x_test in enumerate(X_test):
            pred_class = clf.predict(x_test.reshape(1, -1))
            classifications.append(pred_class)
            true_class.append(y_test[i])

        confusion_1 = confusion_matrix(true_class, classifications)
        print_and_write4("SVC Linear confusion")
        print_and_write4(confusion_1)

        accuracy_1 = accuracy(confusion_1)
        print_and_write4("SVC Linear accuracy")
        print_and_write4(accuracy_1)
        write_to_csv_part4(str(accuracy_1) + ",")
        best = 1
        best_acc = accuracy_1

        precision_1 = precision(confusion_1)
        print_and_write4("SVC Linear precision")
        print_and_write4(precision_1)

        recall_1 = recall(confusion_1)
        print_and_write4("SVC Linear recall")
        print_and_write4(recall_1)

        write_to_csv_4(1, accuracy_1, recall_1, precision_1, confusion_1)
        
        # B) RADIAL

        print_and_write4("\n\nRadial SVC")

        clf = SVC(kernel='rbf', gamma=2)
        clf.fit(X_train, y_train)


        classifications = []
        true_class = []


        print_and_write4("Radial test data...")
        for i, x_test in enumerate(X_test):
            pred_class = clf.predict(x_test.reshape(1, -1))
            classifications.append(pred_class)
            true_class.append(y_test[i])

        confusion_2 = confusion_matrix(true_class, classifications)
        print_and_write4("SVC Radial confusion")
        print_and_write4(confusion_2)

        accuracy_2 = accuracy(confusion_2)
        print_and_write4("SVC Radial accuracy")
        print_and_write4(accuracy_2)
        write_to_csv_part4(str(accuracy_2) + ",")

        if(accuracy_2 > best_acc):
            best_acc = accuracy_2
            best = 2

        precision_2 = precision(confusion_2)
        print_and_write4("SVC Radial precision")
        print_and_write4(precision_2)

        recall_2 = recall(confusion_2)
        print_and_write4("SVC Radial recall")
        print_and_write4(recall_2)


        write_to_csv_4(2, accuracy_2, recall_2, precision_2, confusion_2)

        # C) RandomForestClassifier

        print_and_write4("\n\nRandomForestClassifier")

        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        clf.fit(X_train, y_train)


        classifications = []
        true_class = []



        print_and_write4("RFC test data...")
        for i, x_test in enumerate(X_test):
            pred_class = clf.predict(x_test.reshape(1, -1))
            classifications.append(pred_class)
            true_class.append(y_test[i])

        confusion_3 = confusion_matrix(true_class, classifications)
        print_and_write4("RFC confusion")
        print_and_write4(confusion_3)

        accuracy_3 = accuracy(confusion_3)
        print_and_write4("RFC accuracy")
        print_and_write4(accuracy_3)
        write_to_csv_part4(str(accuracy_3) + ",")

        if(accuracy_3 > best_acc):
            best_acc = accuracy_3
            best = 3

        precision_3 = precision(confusion_3)
        print_and_write4("RFC precision")
        print_and_write4(precision_3)

        recall_3 = recall(confusion_3)
        print_and_write4("RFC recall")
        print_and_write4(recall_3)


        write_to_csv_4(3, accuracy_3, recall_3, precision_3, confusion_3)


        # D) MLPClassifier

        print_and_write4("\n\nMLPClassifier")

        clf = MLPClassifier(alpha=0.05)
        clf.fit(X_train, y_train)


        classifications = []
        true_class = []

        print_and_write4("MLP test data...")
        for i, x_test in enumerate(X_test):
            pred_class = clf.predict(x_test.reshape(1, -1))
            classifications.append(pred_class)
            true_class.append(y_test[i])

        confusion_4 = confusion_matrix(true_class, classifications)
        print_and_write4("MLP confusion")
        print_and_write4(confusion_4)

        accuracy_4 = accuracy(confusion_4)
        print_and_write4("MLP accuracy")
        print_and_write4(accuracy_4)
        write_to_csv_part4(str(accuracy_4) + ",")

        if(accuracy_4 > best_acc):
            best_acc = accuracy_4
            best = 4

        precision_4 = precision(confusion_4)
        print_and_write4("MLP precision")
        print_and_write4(precision_4)

        recall_4 = recall(confusion_4)
        print_and_write4("MLP recall")
        print_and_write4(recall_4)


        write_to_csv_4(4, accuracy_4, recall_4, precision_4, confusion_4)


        # E) AdaBoostClassifier

        print_and_write4("\n\nAdaBoostClassifier")

        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)


        classifications = []
        true_class = []


        print_and_write4("ABC test data...")
        for i, x_test in enumerate(X_test):
            pred_class = clf.predict(x_test.reshape(1, -1))
            classifications.append(pred_class)
            true_class.append(y_test[i])

        confusion_5 = confusion_matrix(true_class, classifications)
        print_and_write4("ABC confusion")
        print_and_write4(confusion_5)

        accuracy_5 = accuracy(confusion_5)
        print_and_write4("ABC accuracy")
        print_and_write4(accuracy_5)
        write_to_csv_part4(str(accuracy_5) + ",")

        if(accuracy_5 > best_acc):
            best_acc = accuracy_5
            best = 5

        precision_5 = precision(confusion_5)
        print_and_write4("ABC precision")
        print_and_write4(precision_5)

        recall_5 = recall(confusion_5)
        print_and_write4("ABC recall")
        print_and_write4(recall_5)


        write_to_csv_4(5, accuracy_5, recall_5, precision_5, confusion_5)



        iBest = best

    return (X_train, X_test, y_train, y_test,iBest)




    
if __name__ == "__main__":

    # TODO : complete each classification experiment, in sequence.
   
   
    data_array = np.load(args.input)
    
    iBest = 4 
    '''
    # 3.1
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
   ''' 


    iBest = 4 
    
    # 3.2
        
    
    for data_use in [250, 1250, 2500, 3750, 5000]:
        X1 = data_array[0:data_use, 0:173]
        y1 = data_array[0:data_use, 173]

        X2 = data_array[10000:10000+data_use, 0:173]
        y2 = data_array[10000:10000+data_use, 173]

        X3 = data_array[20000:20000+data_use, 0:173]
        y3 = data_array[20000:20000+data_use, 173]

        X4 = data_array[30000:30000+data_use, 0:173]
        y4 = data_array[30000:30000+data_use, 173]
        
        best_acc = 0
        best = 0

        X_train = np.concatenate((X1, X2, X3, X4))
        y_train = np.concatenate((y1, y2, y3, y4))

        X1 = data_array[data_use:10000, 0:173]
        y1 = data_array[data_use:20000, 173]

        X2 = data_array[10000+data_use:20000, 0:173]
        y2 = data_array[10000+data_use:20000, 173]

        X3 = data_array[20000+data_use:30000, 0:173]
        y3 = data_array[20000+data_use:30000, 173]

        X4 = data_array[30000+data_use:40000, 0:173]
        y4 = data_array[30000+data_use:40000, 173]

        X_test = np.concatenate((X1, X2, X3, X4))
        y_test = np.concatenate((y1, y2, y3, y4))

        class32(X_train, X_test, y_train, y_test, iBest)
    

    
     # 3.3
    
    X = data_array[:, 0:173]
    y = data_array[:, 173]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #X_1k = X[0:1000]
    #y_1k = y[0:1000]

    data_use = 250

    X1 = data_array[0:data_use, 0:173]
    y1 = data_array[0:data_use, 173]

    X2 = data_array[10000:10000+data_use, 0:173]
    y2 = data_array[10000:10000+data_use, 173]

    X3 = data_array[20000:20000+data_use, 0:173]
    y3 = data_array[20000:20000+data_use, 173]

    X4 = data_array[30000:30000+data_use, 0:173]
    y4 = data_array[30000:30000+data_use, 173]
    
    best_acc = 0
    best = 0

    X_1k = np.concatenate((X1, X2, X3, X4))
    y_1k = np.concatenate((y1, y2, y3, y4))

    for top_k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(f_classif, k=top_k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_

        csv_3_3.write("32k " + str(top_k) + ",")
        for i in pp:
            csv_3_3.write(str(i) + ",")
        csv_3_3.write("\n")

        selector = SelectKBest(f_classif, k=top_k)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_

        csv_3_3.write("1k " + str(top_k) + ",")
        for i in pp:
            csv_3_3.write(str(i) + ",")
        csv_3_3.write("\n")

    data_use = 500
 
    X1 = data_array[0:data_use, 0:173]
    y1 = data_array[0:data_use, 173]

    X2 = data_array[10000:10000+data_use, 0:173]
    y2 = data_array[10000:10000+data_use, 173]

    X3 = data_array[20000:20000+data_use, 0:173]
    y3 = data_array[20000:20000+data_use, 173]

    X4 = data_array[30000:30000+data_use, 0:173]
    y4 = data_array[30000:30000+data_use, 173]
    

    X_2k = np.concatenate((X1, X2, X3, X4))
    y_2k = np.concatenate((y1, y2, y3, y4))


    print("Shape of X_2k: " + str(X_2k.shape))
    print("Shape of y_2k: " + str(y_2k.shape))
 
    X_2k_best = SelectKBest(f_classif, k=5).fit_transform(X_2k, y_2k)
    X_best = SelectKBest(f_classif, k=5).fit_transform(X, y)


    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X_best, y, test_size=0.2, random_state=42)

    print("Shape of X_train: " + str(X_train_1.shape))
    print("Shape of y_train: " + str(y_train_1.shape))



    X_1k_train, X_1k_test, y_1k_train, y_1k_test = train_test_split(
        X_2k_best, y_2k, test_size=0.5, random_state=42)


    class33(X_train_1, X_test_1, y_train_1, y_test_1, iBest, X_1k_train, X_1k_test, y_1k_train, y_1k_test)
    
    # class34
    
    class34(args.input, iBest)
    
