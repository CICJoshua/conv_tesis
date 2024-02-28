""" 
-------------------------------
    Autor: Joshua Guerrero
    Fecha: 8 de abril, 2022
-------------------------------
    Descripción:

        FactoryPlots -------> Clase para crear listas de variables de dimensión 'n', dados los atributos pasados en los argumentos
                                del constructor de la clase.
"""

import json
import pandas
""" import matplotlib.pyplot as plot
 """
from tqdm import tqdm
import sys
import os

class FactoryPlots:
    def __init__(self, path) -> None:
        self.__path  = path
        self.__tree = None
        self.__file = open(self.__path)
        self.__attr = None
        """ self.__plot_width  = int(750)
        self.__plot_height = int(self.__plot_width//1.2)
        self.__plot_options = hv.Options(width=self.__plot_width, height=self.__plot_height, xaxis=None, yaxis=None) """

    def __str__(self) -> str:
        return self.__tree

    def __del__(self) -> str:
        pass
    

    def getAttr(self):
        self.__tree = json.load(self.__file)
        temp = [i for i in iter(self.__tree)]
        temp2 = temp[0]
        self.__attr = [attr for attr in iter(self.__tree[temp2])]
        return self.__attr

    def oneDVariable(self, attr, icd):
        __x = []

        try:
            for notas in self.__tree:
                temp = self.__tree[notas]
                if icd in temp:
                    temp2 = self.__tree[notas][icd]

                    __x.append(temp2)
                    continue
            return __x
        except:
            self.__file.close()
            print("Error en oneDPlots...")
    
    def oneDPlot(self, x, attr):
        self.__variable = x
        self.__attr_scope = attr
        self.__series = pandas.Series(self.__variable,name=f"{self.__attr_scope}", copy="false")
        self.__data_frame = pandas.DataFrame()
        
        
        __dirname = os.getcwd()

        __file_dic = f"{__dirname}\\graphs\\json\\"
        __file_name = f"count.json"
        __file_json = __file_dic + __file_name
        temp = os.listdir(__file_dic)
        if not __file_name in os.listdir(__file_dic) or len(open(__file_json).readlines()) <= 0:
            open(__file_dic + __file_name, 'a').close()
            
            self.__data = {}
            self.__set_data = set()
            
            for attr in tqdm(self.__variable):
                self.__set_data.add(attr)
            for attr in tqdm(self.__set_data):
                self.__count = 0
                for data in self.__series.values:
                    if attr == data:
                        self.__count += 1
                    else:
                        continue
                self.__data[attr] = self.__count
            
            print("--------------")
            print("Creando archivo")
            print("--------------")
            with open(__file_dic + __file_name, 'w')as out_file:
                out_file.write(json.dumps(self.__data))
                out_file.close()
                return True
        else:
            print("--------------")
            print("Ya hay archivo")
            print("--------------")
            
            print(f"Desea borrar el archivo existente: {__file_name}")
            print("S: Sí")
            print("N: No")
            for line in input():
                if "s" in line.lower():
                    os.remove(__file_json)
                    break
                elif "n" in line.lower():
                    return True
                else:
                    print("Favor de introducir un valor correcto...")
        
        """ return self.__series """
