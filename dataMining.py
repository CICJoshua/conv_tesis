""" 
    dataMining.py ----> Esta función nos permitirá visualizar la estadística de la base de datos 
                        de notas clínicas.
    

    Clases y métodos públicos:
    
        1. dataFrame: Clase para retornar un dataFrame con las propiedades siguientes...

                - getDataframe: Método que regresa una representación en tabla de los datos que se tienen en el json
    
        2. statsDesc: Clase para sacar estadística descriptiva de la base de datos de notas clínicas

                - getModeGraph: Método para retornar una gráfica de la Moda (aritmética) del atributo (ICD), etiquetas
                                de acuerdo al estandar clínico ICD 10





"""
from turtle import st

from tqdm import tqdm
from parts.dataFrame import DataFrame
from parts.statsDesc import StatsDesc
import os
""" import matplotlib.pyplot as plt
 """
def main():
    __dirname = os.getcwd()
    __list_files = os.listdir(os.path.join(__dirname, "datos"))
    __file = [file for file in tqdm(__list_files) if "json" in file][0]
    
    __path = f"{__dirname}\\datos\\{__file}"
    """ 
    dataframe = DataFrame()
    data = dataframe.getDataframe(__path)
    datos = data.head() 
    """
    stat_desc = StatsDesc(__path)
    
    __data_frame = stat_desc.getDataFrame()
    print("Debugger")
    """ del dataframe """

if __name__ == "__main__":
    main()