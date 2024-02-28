""" 

    Autor: Joshua Guerrero

    Descripción:

        statsDesc ------->  Clase para obtener un análisis estadístico descriptivo de las notas clínicas dermatológicas.

            1. getCatPlot: Gráfica para visualizar datos categóricos vs una variable numérica (Count)

"""
from tqdm import tqdm
from parts.dataFrame import DataFrame

from tkinter.font import names
from turtle import color
import numpy

from pyparsing import col
from graphs.factoryPlots import FactoryPlots
""" import matplotlib.pyplot as multiple_graph
 """
""" import seaborn
 """
import os
import json
import pandas
from tqdm import tqdm
import seaborn

class StatsDesc:
    
    def __init__(self, path) -> None:
        self.__path = path

    def __del__(self) -> None:
        print("Objeto destruido...")

    def __str__(self) -> str:
        return self.__path

    def getDataFrame(self):
        
        """ Atravesamos el árbol de notas clínicas para obtener las variables 'x' 'y' """

        """ x, y = twodPlot(...keys) """

        """ factoryPlot = FactoryPlots(self.__path)
        self.__attr = factoryPlot.getAttr()
        self.__name_attr = "dx1"
        self.__x = factoryPlot.oneDVariable(self.__attr, self.__name_attr)
        self.__tags = factoryPlot.oneDPlot(self.__x, self.__name_attr) """
        
        """ Leo el archivo JSON que contiene un diccionario de python """
        self.__dirname = os.getcwd()
        self.__file_dic = f"{self.__dirname}\\datos\\"
        self.__file_name = f"pacientes.json"
        self.__file_json = self.__file_dic + self.__file_name
        
        """ Genero una instancia de un objeto DatraFrame """
        self.__data_frame = DataFrame()
        self.__frame_data = self.__data_frame.getDataframe(self.__file_json)

        
        
        """ Generamos una columna con las etiquetas de los diagnósticos 
            de las notas clíncias """
        self.__tags = self.__frame_data.groupby(["dx1"])["dx1"].count()
        self.__dates = self.__frame_data.groupby(["fecha"])["fecha"].count()
        print(pandas.DatetimeIndex(self.__frame_data["fecha"]))
        """ Creamos 2 listas que contienen la frequencia y las etiquetas del objeto dataframe"""
        self.__freq_arr = []
        self.__freq_tags = []
        self.__dates_list = []
        for row in tqdm(self.__frame_data["dx1"]):
            if row in self.__tags:
                self.__freq_arr.append(self.__tags[row])
                self.__freq_tags.append(row)
        self.__frame_data["Count"] = self.__freq_arr
        
        """ Convertimos las listas en un objeto serie """
        self.__df = pandas.Series(self.__freq_tags)
        self.__df2 = pandas.Series(self.__freq_arr)
        
        """ Obtenemos un objeto dataframe de los objetos Series """
        self.df3 = pandas.DataFrame({'freq':self.__df2,"tags":self.__df})
        
        "retornamos un objeto DataFrame de dos columnas (freq: frecuencia, tags: etiquetas de diagnóstico ICD"
        
        self.__freq_tags = self.df3.drop_duplicates().sort_values(by=['freq'],ascending=False)
        
        print(self.__freq_tags["freq"].head())
        print(self.__freq_tags["tags"].head())
        
        """ TODO: obtener una lista de de n, en donde n es la longitud
            de la lista de las etiquetas (tags) en el DF...
        """
        
        return seaborn.barplot(x=
                                    self.__freq_tags["tags"].head(10),
                                y=
                                    self.__freq_tags["freq"].head(10),
                                data=self.__freq_tags
                               )
        

"""         return graph.displot(x="dx1",y=self.__frame_data["cat"],data=self.__frame_data)
 """         
        