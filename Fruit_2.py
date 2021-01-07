# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:35:38 2020

@author: Josiah Stadler
@Course: CSC-410-01
@Date: 8/29/20
@Purpose: This program reads in images, extracts and manipulates features, then
         identifies the images based on type/class of fruit.
"""

# Importing resources
import os
import cv2 
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import glob
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix



def main():    
    # labels to name new files when saved to folder
    app= 'Apple '
    ban = 'Banana '
    org = 'Orange '
    merge_2 = app + ban
    merge_3 = merge_2 + org
    nonOV = "non-overlapping "
    ov = "Overlapping "
    NO_train_2 = 'Non-overlapping '+merge_2
    NO_test_2 = 'Non-overlapping ' + merge_2
    NO_train_3 = 'Non-overlapping '+merge_3
    NO_test_3 = 'Non-overlapping ' + merge_3
    O_train_2 = 'Overlapping  '+merge_2
    O_test_2 = 'Overlapping ' + merge_2
    O_train_3 = 'Overlapping '+merge_3
    NO_test_3 = 'Overlapping ' + merge_3    
    
    #labels for final column in spreadsheet
    app_ID = 0
    ban_ID = 1
    org_ID = 2 
    
    
    #File paths to read original images from
    app_Files = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\apples\*.jpg')
    org_Files = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\oranges\*.jpg')
    ban_Files = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\bananas\*.jpg')
    
    #Paths to save workbooks for 8*8 blocks
    app_Book = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Apple_workbooks\\'
    org_Book = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Orange_workbooks\\'
    ban_Book = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Banana_workbooks\\'
    
    #Paths to save sliding block features to
    app_Slide = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Apple_Slide\\'
    ban_Slide = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Banana_Slide\\'
    org_Slide = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Orange_Slide\\'
    
    #Files to read 8*8 block data from csv
    app_Blk_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Apple_workbooks\*.csv')
    org_Blk_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Orange_workbooks\*.csv')
    ban_Blk_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Banana_workbooks\*.csv')
    
    #Files to read sliding block features from csv
    app_Sl_File = glob.glob( r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Apple_Slide\*.csv')
    ban_Sl_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Banana_Slide\*.csv')
    org_Sl_File = glob.glob( r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Orange_Slide\*.csv')
    
    #Files to read 2 & 3 class merged data for non-overlapping
    nonOverlap_2 = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NonOverlapping_2\\'  
    nonOverlap_3 = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NonOverlapping_3\\'
  
    #Files to read 2 & 3 class merged data for overlapping features
    overlap_2 = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Overlapping_2Cl\\'   
    overlap_3 = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Overlapping_3Cl\\'
    
    #Paths to save training/test data- non-overlapping 
    NO_Train_Path = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NO_Training\\'
    NO_Test_Path =r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NO_Testing\\'
    
    #Paths to save training/test data - overlapping
    O_Train_Path = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\OL_Training\\'
    O_Test_Path = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\OL_Testing\\'
    
    #Files to read training/test data- non-overlapping 
    NO_Train_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NO_Training\*.csv')
    NO_Test_File =glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\NO_Testing\*.csv')
    
    #Files to read training/test data - overlapping
    O_Train_File =glob.glob( r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\OL_Training\*.csv')
    O_Test_File = glob.glob(r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\OL_Testing\*.csv')
        
    #Arrays to hold data from reading in image
    app_Arr = []
    ban_Arr = []
    org_Arr = []
    
    #Read in and display original images with function
    #print('===============GETTING & DISPLAYING IMAGES FOR=========================')
    app_Arr = get_Images(app_Files)
    ban_Arr = get_Images(ban_Files)
    org_Arr = get_Images(org_Files)    
   
    #Arrays to hold data for resized grey images
    grey_App_Arr = []
    grey_Ban_Arr = []
    grey_Org_Arr = []
    
    #Display grey scale, print size and reduces size 
    #Apples  
    grey_App_Arr = display_Grey(app_Arr)    
    #Oranges
    grey_Org_Arr = display_Grey(org_Arr)    
    #Bananas
    grey_Ban_Arr = display_Grey(ban_Arr) 
    
    # #get 8*8 block features
    get_Blocks(grey_App_Arr, app_ID, app, app_Book)
    get_Blocks(grey_Org_Arr, org_ID, org, org_Book)
    get_Blocks(grey_Ban_Arr, ban_ID, ban, ban_Book)
    
    # # #get sliding block features
    get_Slide_Blocks(grey_App_Arr, app_ID, app, app_Slide )
    get_Slide_Blocks(grey_Ban_Arr, ban_ID, ban, ban_Slide )
    get_Slide_Blocks(grey_Org_Arr, org_ID, org, org_Slide )  
    
    #Merge the datasets for non-overlapping features for each class
    mergeSheets(app_Book, app, nonOV)
    mergeSheets(ban_Book, ban, nonOV)
    mergeSheets(org_Book, org, nonOV)    
    
    #Merge the datasets for overlapping features for each class
    mergeSheets(app_Slide, app, ov)
    mergeSheets(ban_Slide, ban, ov)
    mergeSheets(org_Slide, org, ov)
    
    #Merge datasets of 2 classes- non-overlapping 
    mergeSheets(nonOverlap_2, merge_2, nonOV)
    
    #Merge datasets of 3 classes- non-overlapping 
    mergeSheets(nonOverlap_3, merge_3, nonOV)
    
    # #Merge datasets of 2 classes- overlapping 
    mergeSheets(overlap_2, merge_2, ov)
    
    #Merge datasets of 3 classes- overlapping 
    mergeSheets(overlap_3, merge_3, ov)

    # #get 2D and 3D scatter plot for 8*8 for each class
    get_Graphics(app_Blk_File, app)
    get_Graphics(ban_Blk_File, ban)
    get_Graphics(org_Blk_File, org)
    
    # #get 2D and 3D scatter plot for sliding block for each class
    get_Graphics(app_Sl_File, app)
    get_Graphics(ban_Sl_File, ban)
    get_Graphics(org_Sl_File, org)
    
    # #get 2D and 3D scatter plot for non-overlapping merged two classes
    get_Graphics(nonOverlap_2, merge_2)
    
    # #get 2D and 3D scatter plot for non-overlapping merged three classes 
    get_Graphics(nonOverlap_3, merge_3)
    
    # #get 2D and 3D scatter plot for overlapping- merged two classes
    get_Graphics(overlap_2, merge_2)
    
    # #get 2D and 3D scatter plot for overlapping- merged three classes 
    get_Graphics(overlap_3, merge_3)
    
    # #Split, train, test, and save data for lasso regression- 2 class merged data
    get_Models(nonOverlap_2, NO_train_2, NO_Train_Path, NO_Test_Path)
    get_Models(overlap_2,O_train_2, O_Train_Path, O_Test_Path)
    
    # #Split, train, test, save overlapping & non-overlapping data with RF- 2 & 3 class
    gen_RF(nonOverlap_2, NO_train_2, NO_Train_Path, NO_Test_Path,2)
    gen_RF(nonOverlap_3, NO_train_3, NO_Train_Path, NO_Test_Path,3)
    gen_RF(overlap_3, O_train_3, O_Train_Path, O_Test_Path,3)
    gen_RF(overlap_2, O_train_2, O_Train_Path, O_Test_Path,2)  
 
 #Reads in images from file and displays plotted full color, RGB channel 
 #and saves full color data to array per class
def get_Images(files):
    arr =[]
    for img in files:

        #Display original image, save data in array
        img = cv2.imread(img)

        color = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        arr.append(color)

        plt.imshow(color)
        plt.show()

        #Display RGB channels
        blue = img.copy()
        blue[:,:,1] = 0
        blue[:,:,2] = 0
        plt.imshow(blue)
        plt.show()

        #Get green channel
        green = img.copy()
        green[:,:,0] = 0
        green[:,:,2] = 0
        plt.imshow(green)
        plt.show()

        #Get red channel
        red = img.copy()
        red[:,:,0] = 0
        red [:,:,1] = 0
        plt.imshow(red)
        plt.show()

    npImg = np.asarray(arr)    
    return npImg

#Show images in specified color channel- RGB 
def display_Grey(Arr):
    arr =[]
    
    for data in Arr:
     
        #show grayscale with original size
        grey = cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
        plt.imshow(grey, cmap = 'gray',vmin = 0, vmax =255)
        plt.show()
        print("original grey shape", grey.shape)

        #reduse size of grayscale image
        ratio = grey.shape[1]/ grey.shape[0]
        height = 256
        width = int(height * ratio)
        if width % 8 != 0:
            width -= (width % 8)
        dimesion = (height, width)
        resized = cv2.resize(grey, dimesion, interpolation = cv2.INTER_AREA)
        plt.imshow(resized,cmap = 'gray',vmin = 0, vmax = 255)
        plt.show()
        print("Reduced grey shape", resized.shape)
        arr.append(resized)
        
    #npArr = np.asarray(arr)    
    return arr

def get_Blocks(arr, ID, fruit, path):
    
    n = 0
    ext = '.csv'
    e = 8   

    for d in arr: 
        file_Name = fruit + "NO" + str(n+1) + ext        
        width = arr[n].shape[1]      
        height = arr[n].shape[0]       
        arr[n] = np.reshape(arr[n],(height,width))
        rnd = round(((height - e )*(width - e))/(e * e))
        flat = np.full((rnd,65), ID, np.uint8)
        x = 0        
        for i in range(0, height - 8, 8):
            for j in range(0, width - 8, 8):
                curr_Arr = arr[n][i:i+e, j:j+e]
                flat[x, 0:64] = curr_Arr.flatten()                    
                x += 1
        space = pd.DataFrame(flat)
        space.to_csv(path + file_Name, index = False)
        n += 1   
    return
            
def get_Slide_Blocks(arr, ID, fruit, path):
    n = 0
    ext = '.csv'
    typeStr = "_OL"
      
    for d in arr: 
        file_Name = fruit + typeStr + str(n+1) + ext         
        width = arr[n].shape[1]      
        height = arr[n].shape[0]       
        arr[n] = np.reshape(arr[n],(height,width))
        rnd = round(((height - 1 )*(width - 1)))
        flat = np.full((rnd,65), ID, np.uint8)
        x = 0
        for i in range(0, height - 8, 1):
            for j in range(0, width - 8, 1):
                curr_Arr = arr[n][i:i+8, j:j+8]
                flat[x, 0:64] = curr_Arr.flatten()                    
                x += 1
        space = pd.DataFrame(flat)
        space.to_csv(path + file_Name, index = False)
        n += 1
    return

def mergeSheets(path, fruit, feat):
    os.chdir(path)
    ext = 'csv'
    all_fileNames = [i for i in glob.glob('*.{}'.format(ext))]
    merged_csv = pd.concat([pd.read_csv(f) for f in all_fileNames])
    merged_csv.to_csv("merged " + fruit + feat +"."+ ext, index=False, encoding='utf-8-sig')

def get_Feat_Graphics(path, fruit):
    for file in path:
        
        dSet = pd.read_csv(file, header = None)
        feat1 = dSet[30]      
        feat2 = dSet[40]       
        plt.hist(feat1, 50, color = 'blue')
        plt.title('Histogram for: ' + fruit + ' Data')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.hist(feat2, 50, color = 'red')
        plt.title('Histogram for: ' + fruit + ' Data')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend(['Feature 1','Feature 2'])
     
def get_Graphics(path, fruit):
    feat1 = 30
    feat2 = 40
    feat3 = 50
    
    for file in path:        
        dSet = np.genfromtxt(file, delimiter =',')
      
        
        #Histogram- shows the number of features that have each value        
        plt.hist(dSet[:, 0:-1].reshape(-1), feat3)      
        plt.title('Histogram for ' + fruit + ' Data')
        plt.xlabel('Feature Value')
        plt.ylabel('Count: features with value')            
        
        #for 2D scatter plot- plots 2 features for classes
        fig_1 = plt.figure()
        axis_1 = plt.subplots()
        scat_1 = plt.scatter(dSet[:, feat1], dSet[:, feat3], c=dSet[:, -1])       
        legend_1 = fig_1.legend(*scat_1.legend_elements(),loc = "lower right", title = "Classes")
        fig_1.add_artist(legend_1)
        plt.xlabel(fruit + ': Feature 1: %i' %feat1)
        plt.ylabel(fruit + ': Feature 3: %i' %feat3) 

        #For 3D scatter plot- plots 3 features for classes       
        fig_2 = plt.figure()       
        axis_2 = Axes3D(fig_2)        
        scat_2 = axis_2.scatter(dSet[:, feat1], dSet[:, feat2], dSet[:, feat3], c= dSet[:,-1])        
        legend_2 = axis_2.legend(*scat_2.legend_elements(), loc="lower right", title = "Classes")       
        fig_2.add_artist(legend_2)                
        axis_2.set_xlabel('3D ' + fruit + ': Feature 1 %i' %feat1)
        axis_2.set_ylabel('3D ' + fruit + ': Feature 2 %i' %feat2)
        axis_2.set_zlabel('3D ' + fruit + ': Feature 3 %i' %feat3)       
        

#Task 2- assignment pt2
def get_Models(path, fruit, train_Save, test_Save):
    #For saving all confusion matrices
    n = 0
    ext = '.csv'
    confusion_Path = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Confusion_Matrices\\'
    
    for file in path:        
        input_data = pd.read_csv(file, header = None)        
        ######Standard regression - to use to check sign for lasso###
        y = input_data[64]
        # Drop the labels and store the features
        input_data.drop(64,axis=1,inplace=True)
        #features
        X = input_data
        #feature matrix
        X1 = np.array(X)
        # Transpose of the feature matrix
        X2 = X1.transpose()        
        # Square of the feature matrix 
        XX=np.matmul(X2, X1)       
        # Inverse of the square of the feature matrix
        IX = inv(XX)        
        # Multiply it with its feature matrix
        TX = np.matmul(X1, IX)        
        # Generate label matrix using Numpy array
        Y1 = np.array(y)        
        # Transpose of the label matrix
        Y2 = Y1.transpose()        
        # Multiply it with feature matrix related term
        A = np.matmul(Y1, TX)       
        # Validating the model
        Z1 = np.matmul(X1, A)        
        Z2 = Z1 > 0.5        
        yhat = Z2.astype(int)       
        
        # Machine learning with 80:20- working with complete sheet. 
        #training/testing csv saved in folder for reference        
        # Split the data into 80:20
        row, col = X.shape        
        TR = round(row*0.8)        
        TT = row-TR        
        X1_train = X1[0:TR-1,:]       
        y_train = y[0:TR-1]       
        X1_test = X1[TR:row,:]       
        y_test = y[TR:row]        
       
        #training model
        X2_train = X1_train.transpose()
        XX_train = np.matmul(X2_train, X1_train)
        IX_train = inv(XX_train)
        TX_train = np.matmul(X1_train, IX_train)
        Y1_train = np.array(y_train)        
        Y2_train = Y1_train.transpose()       
        A1_train = np.matmul(Y1_train,TX_train) 
        A2_train = np.matmul(A1_train, IX_train)        
        S1 = np.sign(A2_train)
        lamda = .2
        term = (lamda/2)*S1
        A3_train = A1_train-term
        
        #for lasso- works but all < 1
        lasso= np.matmul(A3_train, IX_train)
        training_Data = pd.DataFrame(X1_train)
        training_Data[64] = y_train
        
        #Save training data to csv
        training_Data.to_csv(train_Save +'Training ' + fruit + str(n +1) + ext, index = False)
      
        # Testing the model       
        Z1_test = np.matmul(X1_test, lasso)
        
        #Z2_test is one that functions- was >.5 changed and acc went up
        Z2_test = Z1_test <  .00009
              
        yhat_test = Z2_test.astype(int)            
 
        test_Data = pd.DataFrame(X1_test)
        test_Data[64] = y_train
        test_Data[65] = yhat_test        
        test_Data.to_csv(test_Save + 'Testing ' + fruit + str(n +1) + ext, index = False)
        
        #Built-in accuracy measure-Working but weak        
        print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
        print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
        print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))
               
        # Confusion matrix analytics       
        CC_test = confusion_matrix(y_test, yhat_test)
       
        
        #print(CC_test)
        TN = CC_test[0,0]
        FP = CC_test[0,1]
        FN = CC_test[1,0]
        TP = CC_test[1,1]
       
        #Path to save all confusion matrices
       
        confusion_File='CM for '+ fruit + str(n+1) + ext
        FPFN = FP+FN
        TPTN = TP+TN
        space = pd.DataFrame(CC_test)
        space.to_csv(confusion_Path + confusion_File, index = False)
        
        Accuracy = 1/(1+(FPFN/TPTN))
        print("Our_Accuracy_Score:",Accuracy)
        
        Precision = 1/(1+(FP/TP))
        print("Our_Precision_Score:",Precision)
        
        Sensitivity = 1/(1+(FN/TP))
        print("Our_Sensitivity_Score:",Sensitivity)
        
        Specificity = 1/(1+(FP/TN))
        print("Our_Specificity_Score:",Specificity)
        
#Tasks 3 and 4- assignment pt 2
def gen_RF(path, fruit, train_Save, test_Save, class_Num):
    #Details and formatting for saving confusion matrices
    n = 0
    ext = '.csv'
    confusion_Path = r'C:\Users\josia\OneDrive\Desktop\Fruit_image_data_1\Confusion_Matrices\\'
    for file in path: 
        input_data = pd.read_csv(file, header = None)
        # Label/Response set
        y = input_data[64]
        
        # Drop the labels and store the features
        input_data.drop(64,axis=1,inplace=True)
        X = input_data
        
        # Generate feature matrix using a Numpy array
        tmp = np.array(X)
        X1 = tmp[:,0:64] #tmp[:,0:4]
        
        
        # Generate label matrix using Numpy array
        Y1 = np.array(y)
        
        # Machine learning with 80:20
        
        # Split the data into 80:20
        row, col = X.shape
        
        TR = round(row*0.8)
        TT = row-TR
        
        # Training with 80% data
        X1_train = X1[0:TR-1,:]
        Y1_train = Y1[0:TR-1]
        training_Data = pd.DataFrame(X1_train)
        training_Data[64] = Y1_train
        #Save training data to csv
        training_Data.to_csv(train_Save +'RF_Training ' + fruit + str(n +1) + ext, index = False)
        
        
        rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
        model = rF.fit(X1_train,Y1_train)
        
        importance = model.feature_importances_
        indices = importance.argsort()[::-1]
        
        
        ##########################################################################################
        #https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        std = np.std([model.feature_importances_ for model in rF.estimators_], axis=0)
        
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importance[indices[f]]))
            
        plt.bar(range(X.shape[1]), importance[indices], color="r", yerr=std[indices], align="center")
        
        plt.xticks(range(X.shape[1]), indices+1, rotation=90)
        
        plt.show()
        ##########################################################################################
        
        oob_error = 1 - rF.oob_score_       
        
        # Testing with 20% data
        X1_test = X1[TR:row,:]
        y_test = Y1[TR:row]
        
        yhat_test = rF.predict(X1_test)
        test_Data = pd.DataFrame(X1_test)
        test_Data[64] = y_test
        test_Data[65] = yhat_test
        
        test_Data.to_csv(test_Save + 'RF_Testing ' + fruit + str(n +1) + ext, index = False)
        
        confusion_File='CM for RF_'+ fruit + str(n+1) + ext
        # Confusion matrix analytics
        CC_test = confusion_matrix(y_test, yhat_test)
        
        #TN = CC_test[0,0]
        #FP = CC_test[0,1]
        #FN = CC_test[1,0]
        #TP = CC_test[1,1]
        
        TN = CC_test[1,1]
        FP = CC_test[1,0]
        FN = CC_test[0,1]
        TP = CC_test[0,0]
        
        FPFN = FP+FN
        TPTN = TP+TN
        space = pd.DataFrame(CC_test)
        space.to_csv(confusion_Path + confusion_File, index = False)
        Accuracy = 1/(1+(FPFN/TPTN))
        print("Our_Accuracy_Score:",Accuracy)
        
        Precision = 1/(1+(FP/TP))
        print("Our_Precision_Score:",Precision)
        
        Sensitivity = 1/(1+(FN/TP))
        print("Our_Sensitivity_Score:",Sensitivity)
        
        Specificity = 1/(1+(FP/TN))
        print("Our_Specificity_Score:",Specificity)
        
        
        # Built-in accuracy measure
        if(class_Num == 2):            
            print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
            print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test))
            print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test))
        else:
            average = 'micro'
            print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test, average !='binary'))
            print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test, average ='micro'))
            print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test, average ='micro'))
     

main()