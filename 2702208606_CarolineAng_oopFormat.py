import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import statistics

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class DataMaker:
    def __init__(self, filePath):
        self.filePath = filePath
        self.df = None
        self.x = None
        self.y = None
    
    def loadData(self):
        self.df = pd.read_csv(self.filePath)

    def createXY(self, targetColumn):
        self.x = self.df.drop(targetColumn, axis=1)
        self.y = self.df[targetColumn]

    def removeUnUsedColumn(self, column):
        self.x = self.x.drop(columns=[column], axis=1)


class ModelMaker:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.createModel()
        self.xTrain, self.xTest, self.yTrain, self.yTest, self.yCheck, self.yRfPred, self.xTrainModel, self.xTestModel = [None]*8

    def createModel(self, criteria='gini', randomState = 42):
        self.rfModel = RandomForestClassifier(criterion=criteria, random_state=randomState)
    
    def dataSplitting(self, testSize = 0.2, randomState = 42):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y, test_size=testSize, random_state=randomState)

    def boxplotMaker(self, columns):
        self.xTrain.boxplot(column=columns)
        plt.show

    def getMedianValue(self, column):
        return np.median(self.xTrain[column].dropna())
    
    def getModeValue(self, column):
        return statistics.mode(self.xTrain[column])
    
    def fillNaData(self, column, data):
        self.xTrain[column] = self.xTrain[column].fillna(data)
        self.xTest[column] = self.xTest[column].fillna(data)

    def labelEncoderMaker(self):
        bookingStatsEncoder = LabelEncoder()

        self.yTrain = bookingStatsEncoder.fit_transform(self.yTrain)
        self.yTest = bookingStatsEncoder.transform(self.yTest)
        self.yCheck = bookingStatsEncoder.inverse_transform(self.yTest)

        pkl.dump(bookingStatsEncoder, open('bookingStatsEncode.pkl', 'wb'))

    def oneHotEncoderMaker(self, column, filename):
        encoder = OneHotEncoder()

        dataTrain = self.xTrain[[column]]
        dataTest = self.xTest[[column]]

        dataTrain = pd.DataFrame(encoder.fit_transform(dataTrain).toarray(), columns=encoder.get_feature_names_out())
        dataTest = pd.DataFrame(encoder.transform(dataTest).toarray(), columns=encoder.get_feature_names_out())

        pkl.dump(encoder, open(filename, 'wb'))

        return dataTrain, dataTest

    def binaryEncoderMaker(self, column):
        binaryEncoder = {}
        data = {'Yes' : 1, 'No' : 0}

        for i in column:
            binaryEncoder[i] = data

        pkl.dump(binaryEncoder, open('binaryEncode.pkl', 'wb'))

    
    def combineColumns(self, train, test):
        self.xTrain = self.xTrain.reset_index()
        self.xTest = self.xTest.reset_index()

        trainDf = pd.concat(train, axis=1)
        testDf = pd.concat(test, axis=1)

        self.xTrain = pd.concat([self.xTrain, trainDf], axis=1)
        self.xTest = pd.concat([self.xTest, testDf], axis=1)

    def removeEncodedColumn(self, column):
        self.xTrainModel = self.xTrain.drop(columns = column, axis=1)
        self.xTestModel = self.xTest.drop(columns = column, axis=1)

    def trainModel(self):
        self.rfModel.fit(self.xTrainModel, self.yTrain)

    def makePrediction(self):
        self.yRfPred = self.rfModel.predict(self.xTestModel)
        
    def createReport(self):
        reverseEncoder = pkl.load(open('bookingStatsEncode.pkl', 'rb'))
        self.yRfPred = reverseEncoder.inverse_transform(self.yRfPred)

        print("Classification Report for Random Forest")
        print(classification_report(self.yCheck, self.yRfPred))

    def saveModel(self):
        rfAcc = accuracy_score(self.yCheck, self.yRfPred)
        print("\nRandom Forest Accuracy Score: %.3f" %rfAcc)
        pkl.dump(self.rfModel, open('outputModel.pkl', 'wb'))

        
        

filePath = 'Dataset_B_hotel.csv'
dataMaker = DataMaker(filePath)
dataMaker.loadData()
dataMaker.createXY('booking_status')
dataMaker.removeUnUsedColumn('Booking_ID')

x = dataMaker.x
y = dataMaker.y


modelMaker = ModelMaker(x, y)
modelMaker.dataSplitting()
modelMaker.boxplotMaker(['required_car_parking_space', 'avg_price_per_room'])

carParking = modelMaker.getMedianValue('required_car_parking_space')
avgPrice = modelMaker.getMedianValue('avg_price_per_room')
mealPlan = modelMaker.getModeValue('type_of_meal_plan')

modelMaker.fillNaData('required_car_parking_space', carParking)
modelMaker.fillNaData('avg_price_per_room', avgPrice)
modelMaker.fillNaData('type_of_meal_plan', mealPlan)

modelMaker.labelEncoderMaker()
trainMeal, testMeal = modelMaker.oneHotEncoderMaker('type_of_meal_plan', 'mealPlanEncode.pkl')
trainRoom, testRoom = modelMaker.oneHotEncoderMaker('room_type_reserved', 'roomTypeEncode.pkl')
trainMarket, testMarket = modelMaker.oneHotEncoderMaker('market_segment_type', 'marketSegmentEncode.pkl')
modelMaker.combineColumns(train=[trainMeal, trainRoom, trainMarket], test=[testMeal, testRoom, testMarket])
modelMaker.binaryEncoderMaker(['required_car_parking_space', 'repeated_guest'])

modelMaker.removeEncodedColumn(['index', 'type_of_meal_plan', 'room_type_reserved', 'market_segment_type'])

modelMaker.trainModel()
modelMaker.makePrediction()
modelMaker.createReport()
modelMaker.saveModel()