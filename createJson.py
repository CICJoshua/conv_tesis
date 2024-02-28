import sys
from parts.parseXml import ParseXml

import os
import json


""" os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6" """


def createXml(path, path_json):

    xml_file = ParseXml()
    tree = xml_file.getTree(f'{path}')
    json_object = json.dumps(tree)
    
    with open(path_json, "w") as out_file:
        out_file.write(json_object)
    out_file.close()

if __name__ == "__main__":
    __dirname = os.getcwd()
    __filename = f'{__dirname}/datos/pacientes.xml'
    __file_dic = f"{__dirname}/datos/"
    __file_name = f"pacientes.json"
    __file_json = __file_dic + __file_name
    
    if not __file_name in os.listdir(__file_dic) or len(open(__file_json).readlines()) <= 0:
        open(__file_dic + __file_name, 'a').close()
        
            
        print("--------------")
        print("Creando archivo")
        print("--------------")
        createXml(__filename, __file_json)
    else:
        print("--------------")
        print("Ya hay archivo")
        print("--------------")
        
        print(f"Desea borrar el archivo existente: {__filename}")
        print("S: Sí")
        print("N: No")
        for line in sys.stdin:
            if "s" in line.lower():
                os.remove(__file_json)
                break
            elif "n" in line.lower():
                break
            else:
                print("Favor de introducir un valor correcto...")
    """ __data_frame.genDataFrame(__file_json) """
