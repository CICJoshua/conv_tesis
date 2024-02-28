import json
import os

from tqdm import tqdm


class GenFileJson:
    def __init__(self):
        pass

    def save_json(self, file_name, json_object):


        __dirname = os.getcwd()

        __file_dic = f"{__dirname}\\datos"
        __file_name = file_name
        __file_json = f"{__file_dic}\\{__file_name}"

        if not __file_name in os.listdir(__file_dic) or len(open(__file_json).readlines()) <= 0:
            open(__file_json, 'a').close()
            
            print("--------------")
            print("Creando archivo")
            print("--------------")
            
            with open(__file_json, 'w')as out_file:
                out_file.write(json.dumps(json_object))
                out_file.close()
                return True
        else:
            print("--------------")
            print("Ya hay archivo")
            print("--------------")
            
            print(f"Desea borrar el archivo existente: {file_name}")
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
        
        return