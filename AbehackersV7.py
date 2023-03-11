
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor


def initialize(train_dataset_rute):
    #Se lee el dataset desde el archivo excel donde se encuentra el dataset
    dataset = pd.read_excel(train_dataset_rute, header = 0, thousands=",")
    
    #Eliminación de variables irrelevantes para el análisis
    dataset.drop(['Fecha entrega del Informe', 'Piso', 'Elevador', 'Depósitos', 'Posición', 
              'Número de frentes', 'Método Representado', 'Número de estacionamiento'], axis=1, inplace=True)
    
    dataset = dataset.loc[dataset["Categoría del bien"] != "AVALUOS_TIPOS_INMUEBLE_VEHICULO", :]
    #dataset = dataset.dropna(subset=['Latitud (Decimal)','Longitud (Decimal)'])
    
    dataset ['Área Terreno'] = dataset ['Área Terreno'].astype(float)
    dataset ['Área Construcción'] = dataset ['Área Construcción'].astype(float)
    
    #Codificación de variables categóricas
    encoder = OrdinalEncoder(categories=[[np.nan, 'Malo', 'Regular - Malo', 'Regular', 'Bueno - Regular', 'Bueno', 'Muy bueno', 'En construcción', 'En proyecto']])
    encoder.fit(dataset[['Estado de conservación']])
    dataset['Estado de conservación'] = encoder.transform(dataset[['Estado de conservación']])
    
    categorical_features =['Categoría del bien', 'Estado de conservación', 'Provincia', 'Distrito', 'Departamento']
    for column in categorical_features:   
        l_encoder = LabelEncoder()
        dataset[column] = l_encoder.fit_transform(dataset[column])
    
    #Rellenado de valores nulos
    dataset['Tipo de vía'] = dataset['Tipo de vía'].fillna(dataset['Tipo de vía'].mode()[0])
    dataset['Distrito'] = dataset['Distrito'].fillna(dataset['Distrito'].mode()[0])
    dataset['Provincia'] = dataset['Provincia'].fillna(dataset['Provincia'].mode()[0])
    dataset['Departamento'] = dataset['Departamento'].fillna(dataset['Departamento'].mode()[0])
    dataset['Estado de conservación'] = dataset['Estado de conservación'].fillna(dataset['Estado de conservación'].mode()[0])
    dataset['Categoría del bien'] = dataset['Categoría del bien'].fillna(dataset['Categoría del bien'].mode()[0])
    dataset['Área Terreno'] = dataset['Área Terreno'].fillna(dataset['Área Terreno'].mean())
    dataset[['Edad', 'Área Construcción']] = dataset[['Edad', 'Área Construcción']].fillna(0)
    
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
    
    #separación del dataset en X y y
    target = dataset['Valor comercial (USD)'].copy()
    data = dataset.copy()
    data.drop('Valor comercial (USD)', axis = 1, inplace = True)
    
    #Separación del dataset en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.1, shuffle = True, random_state=0)
    
    #Implementación de un RandomForestRegressor
    model_RFR = RandomForestRegressor(n_estimators=10, random_state = 1)
    model_RFR.fit(X_train, y_train)
    y_pred = model_RFR.predict(X_test)
    print(model_RFR.score(X_test, y_test))
    print(mean_absolute_percentage_error(y_test, y_pred))
    
    return model_RFR

def fit(test_dataset_rute, model_RFR):
    #Se lee el dataset desde el archivo excel donde se encuentra el dataset
    dataset = pd.read_excel(test_dataset_rute, header = 0, thousands=",")
    
    #Eliminación de variables irrelevantes para el análisis
    df_ID = dataset['ID']
    dataset.drop(['ID', 'Fecha entrega del Informe', 'Piso', 'Elevador', 'Depósitos', 'Posición', 
                  'Número de frentes', 'Método Representado', 'Número de estacionamiento'], axis=1, inplace=True)
    
    dataset = dataset.loc[dataset["Categoría del bien"] != "AVALUOS_TIPOS_INMUEBLE_VEHICULO", :]
    
    dataset ['Área Terreno'] = dataset ['Área Terreno'].astype(float)
    dataset ['Área Construcción'] = dataset ['Área Construcción'].astype(float)
    
    #Codificación de variables categóricas
    encoder = OrdinalEncoder(categories=[[np.nan, 'Malo', 'Regular - Malo', 'Regular', 'Bueno - Regular', 'Bueno', 'Muy bueno', 'En construcción', 'En proyecto']])
    encoder.fit(dataset[['Estado de conservación']])
    dataset['Estado de conservación'] = encoder.transform(dataset[['Estado de conservación']])
    
    categorical_features =['Categoría del bien', 'Estado de conservación', 'Provincia', 'Distrito', 'Departamento']
    for column in categorical_features:   
        l_encoder = LabelEncoder()
        dataset[column] = l_encoder.fit_transform(dataset[column])
    
    #Rellenado de valores nulos
    dataset['Tipo de vía'] = dataset['Tipo de vía'].fillna(dataset['Tipo de vía'].mode()[0])
    dataset['Distrito'] = dataset['Distrito'].fillna(dataset['Distrito'].mode()[0])
    dataset['Provincia'] = dataset['Provincia'].fillna(dataset['Provincia'].mode()[0])
    dataset['Departamento'] = dataset['Departamento'].fillna(dataset['Departamento'].mode()[0])
    dataset['Estado de conservación'] = dataset['Estado de conservación'].fillna(dataset['Estado de conservación'].mode()[0])
    dataset['Categoría del bien'] = dataset['Categoría del bien'].fillna(dataset['Categoría del bien'].mode()[0])
    dataset['Área Terreno'] = dataset['Área Terreno'].fillna(dataset['Área Terreno'].mean())
    dataset[['Edad', 'Área Construcción']] = dataset[['Edad', 'Área Construcción']].fillna(0)
    
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
    
    dataset['Latitud (Decimal)'] = dataset['Latitud (Decimal)'].fillna(dataset['Latitud (Decimal)'].mean())
    dataset['Longitud (Decimal)'] = dataset['Longitud (Decimal)'].fillna(dataset['Longitud (Decimal)'].mean())
    
    #separación del dataset en X y y
    X_test = dataset.copy()
    X_test.drop('Valor comercial (USD)', axis = 1, inplace = True)
    
    y_pred = model_RFR.predict(X_test)
    y_pred = [int(y) for y in y_pred]
    
    df = pd.DataFrame(y_pred, index = df_ID, columns = ['Valor comercial (USD)'])
    df.to_excel('output.xlsx')

def main(train_dataset_rute, test_dataset_rute):
    model = initialize(train_dataset_rute)
    fit(test_dataset_rute, model)

print(main('dataset_tasacion_train_vf.xlsx', 'dataset_test_pruebas (1).xlsx'))
