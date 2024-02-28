""" 
    ParseXml ---> Clase que parsea un archivo XML y lo convierte en objeto JSON,
                y lo transforma en un archivo cvs.
    
    Métodos públicos:
                
        getTree: Método que que retorna una estructura de dato árbol. La cual 
        contiene 600k aprox. de notas clínicas.
        
        saveTree: Transforma un objeto en un archivo "x", en donde "x" es un argumento
        que entra como argumento al método.
    
    Métodos privados:

        __readXML: Método que recive un stream de bytes, el cual parsea los inicios
        de "tags" XML <row>, </row>. Crea un Diccionario de Python
        __find: Método que crea una lista de las encodificaciones disponibles en el módulo
                "aliases"
"""

import codecs
from xml.dom.minidom import Element
from attr import attrib
from numpy import NaN
from tqdm import tqdm
from encodings.aliases import aliases
import xml.etree.ElementTree as ET
 
class ParseXml(object):
    def __init__(self):
        self.__tree = NaN
        self.__fileDir = ""
        self.__file = NaN

    def __del__(self):
        pass

    def getTree(self, file):
        self.__fileDir = file
        """ Comentarios """
        print(self.__fileDir)
        __array_enconde = self.__find()
        for __code in __array_enconde:
            if "ut" in __code:
                try:
                    self.__file = open(self.__fileDir, "r", encoding=__code)
                    self.__tree = self.__readXML(self.__file)
                    break
                except:
                    self.__file.close()
                    print(f"Error en la codificación: {__code}")
                    continue
            else:
                continue

        return self.__tree
    def saveTree(self, file):
        print(file)

    def __find(self):
        return [k for k, v in aliases.items()]

        
    
    def __readXML(self, file):
        __start = f'<row>'
        __end = f'</row>'
        __stringObject = ""
        __tree = {}
        __index = 0
        
        for __lines in tqdm(file):

            if("<table" in __lines or "<?xml" in __lines):
                continue
            if(__start in __lines):
                __stringObject += __lines
            elif __end in __lines:
                __stringObject += __lines
                try:
                    """ 
                        Saca los valores del tipo de dato ELement:
                            TODO: Por definir
                     """
                    __child_tree = {}

                    __element = ET.fromstring(__stringObject)
                    for __child in __element:
                        if __child.text:
                            __child_tree[f"{[v for k, v in __child.items()][0]}"] = f"{__child.text}"
                        else:
                            __child_tree[f"{[v for k, v in __child.items()][0]}"] = ""
                            
                except:
                    continue
                __tree[f'nota_{__index}'] = __child_tree
                __stringObject = ""
                __index += 1
            else:
                __stringObject += __lines
        
        return __tree

  
        

    
