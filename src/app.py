'''
Created on Oct 26, 2019

@author: tfu
'''

import pandas as pd
#import matplotlib.lines as mlines
#import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties

def file2matrix():
    df=pd.read_csv(r'D:/Child Protection/datingTestSet.txt',header=None, sep='\t')
    dataSet= df.loc[:,0:2]
    pre_t=pd.DataFrame(df.loc[:,3])
    #data.loc[data['Y'] == 'T'] = 1
    pre_t[pre_t[3]=='didntLike']=1
    pre_t[pre_t[3]=='smallDoses']=2
    pre_t[pre_t[3]=='largeDoses']=3
    return dataSet,pre_t

def trainNorm(dataSet):
    matMax=dataSet.max(0)
    matMin=dataSet.min(0)
    resu=(dataSet-matMin)/(matMax-matMin)
    return resu

def testNorm(testData,dataSet):
    matMax=dataSet.max(0)
    matMin=dataSet.min(0)
    resu=(testData-matMin)/(matMax-matMin)
    return resu

def trainTestSplitor(splitNum,returnMat,classLabelVector):
    trainMat=returnMat.loc[0:splitNum,:]
    trainVecMat=classLabelVector.loc[0:splitNum,:]
    testMat=returnMat.loc[splitNum+1:,:]   
    return trainMat,trainVecMat,testMat

def classify0(inX,normedMat,classLabelVector):
    diffMat=inX-normedMat
    dist=((diffMat**2).sum(axis=1))**0.5
    midRes=pd.concat([dist,classLabelVector], axis=1)
    sorted_Res=midRes.sort_values(by=0)
    return sorted_Res[[3]][0:50].mode()


def runModelTest(i):
    returnMat,classLabelVector = file2matrix()
    trainMat,trainVecMat,testMat=trainTestSplitor(i,returnMat,classLabelVector)
    normedTrainMat=trainNorm(trainMat)
    normedTestMat=testNorm(testMat,trainMat)
        
    TNum=0
    FNum=0
    for i in normedTestMat.index:
        test_row=pd.Series(normedTestMat.loc[i,:])
        predict=classify0(test_row,normedTrainMat,trainVecMat)
        if predict.values[0][0]==classLabelVector.loc[i,3]:
            TNum=TNum+1
        else:
            FNum=FNum+1
#        print('Predict:'+str(predict.values[0][0]),'Truth:'+str(classLabelVector.loc[i,3]))
    
    accuracyRate=TNum*100/(TNum+FNum)
#    print("T:"+str(TNum))
#    print("F:"+str(FNum))
#    print("Predict Accuracy Rate:"+str(accuracyRate)+r"%")
    return accuracyRate

def recordAccutrateRate():    
    for i in range(1,999,10): 
        rate=runModelTest(i)
        print(i,rate)
        
recordAccutrateRate()


