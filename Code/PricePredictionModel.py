import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class PricePredictionModel():
    def __init__(self, n = 10, random_state = 1):
        #Implementación de un RandomForestRegressor
        model_RFR = RandomForestRegressor(n_estimators = n, random_state = random_state)
        self.model = model_RFR
        
    def preprocess(self, dataset):
        #Eliminación de variables irrelevantes para el análisis        
        dataset.drop(['Fecha entrega del Informe', 'Elevador', 'Piso', 'Depósitos', 'Posición', 
                    'Número de frentes', 'Método Representado', 'Moneda principal para cálculos', 'Número de estacionamiento'], axis=1, inplace=True)
        
        #Filtrado de valores no numéricos
        dataset ['Área Terreno'] = dataset ['Área Terreno'].astype(float)
        dataset ['Área Construcción'] = dataset ['Área Construcción'].astype(float)
        dataset ['Latitud (Decimal)'] = dataset ['Latitud (Decimal)'].astype(float)
        dataset ['Laongitud (Decimal)'] = dataset ['Longitud (Decimal)'].astype(float)

        #Codificación de variables categóricas
        encoder = OrdinalEncoder(categories= 'auto')
        encoder.fit(dataset[['Estado de conservación']])
        dataset['Estado de conservación'] = encoder.transform(dataset[['Estado de conservación']])

        categorical_features =['Categoría del bien', 'Calle', 'Provincia', 'Distrito', 'Departamento']
        for column in categorical_features:   
            l_encoder = LabelEncoder()
            dataset[column] = l_encoder.fit_transform(dataset[column])

        #Rellenado de valores nulos
        dataset['Distrito'] = dataset['Distrito'].fillna(dataset['Distrito'].mode()[0])
        dataset['Provincia'] = dataset['Provincia'].fillna(dataset['Provincia'].mode()[0])
        dataset['Departamento'] = dataset['Departamento'].fillna(dataset['Departamento'].mode()[0])
        dataset['Estado de conservación'] = dataset['Estado de conservación'].fillna(dataset['Estado de conservación'].mode()[0])
        dataset['Categoría del bien'] = dataset['Categoría del bien'].fillna(dataset['Categoría del bien'].mode()[0])
        dataset['Área Terreno'] = dataset['Área Terreno'].fillna(dataset['Área Terreno'].mean())
        dataset[['Edad', 'Área Construcción']] = dataset[['Edad', 'Área Construcción']].fillna(0)
        dataset = dataset.query(' `Latitud (Decimal)` < -0.03 & `Latitud (Decimal)` > -18.3522222 & `Longitud (Decimal)` > -81.32638888888889 & `Longitud (Decimal)` < -68.6575')
        
        #Asignación a valores vacíos de Latitud y Longitud
        Distritos2=dataset['Distrito'].unique()
        Distritos2.shape
        DicLon = {}
        DicLat = {}
        for i in range(len(Distritos2)):
            Distritoi = dataset.loc[dataset['Distrito'] == Distritos2[i]]
            a = Distritoi['Latitud (Decimal)'].mean()
            b = Distritoi['Longitud (Decimal)'].mean()
            DicLat[Distritos2[i]]=round(a, 6)
            DicLon[Distritos2[i]]=round(b, 6)
            
        dataset.loc[dataset['Longitud (Decimal)'].isnull(), 'Longitud (Decimal)'] = dataset['Distrito'].map(DicLon)
        dataset.loc[dataset['Latitud (Decimal)'].isnull(), 'Latitud (Decimal)'] = dataset['Distrito'].map(DicLat)

        #Separación del dataset en X y y
        target = dataset['Valor comercial'].copy()
        data = dataset.copy()
        data.drop('Valor comercial', axis = 1, inplace = True)

        #Separación del dataset en datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.1, shuffle = True, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train, y_train,):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test, csv_output = True):
        y_pred = self.model.predict(X_test)
        y_pred = [int(y) for y in y_pred]

        if csv_output:
            df = pd.DataFrame(y_pred, columns = ['Valor comercial'])
            df.to_excel('output.xlsx')
            
        return y_pred
        