""" 

    Script para generar un árbol jerárquico (Archivo JSON)
    
    Programación funcional
    
    -   build_tree: Función que nos regresa un árbol jerárquico, el cual contiene
        una representación de los nodos padre, nodes hijo y sus aristas; 
        relación entre nodos hijo y padre
        
    NOTA: Estructura de datos (JSON)
"""

import json
import os
from tqdm import tqdm
from utils.genFileJson import GenFileJson

tags = {
        "D1" : "Infecciones de la piel y del tejido subcutáneo",
        "D2" : "Trastornos flictenulares",
        "D3" : "Dermatitis y eczema",
        "D4" : "Trastornos papuloescamosos",
        "D5" : "Urticaria y eritema",
        "D6" : "Trastornos de la piel y del tejido subcutáneo relacionados con radiación",
        "D7" : "Trastornos de las faneras",
        "D8" : "Otros trastornos de la piel y del tejido subcutáneo"
    }
""" 
    tabla hash con los siguientes atributos:
    - Grupo
    - Límites entre enfermedades (Min, Max)
"""
groups = {
    "D1" : ["L00", "L08"],
    "D2" : ["L10", "L14"],
    "D3" : ["L20", "L30"],
    "D4" : ["L40", "L45"],
    "D5" : ["L50", "L54"],
    "D6" : ["L55", "L59"],
    "D7" : ["L60", "L75"],
    "D8" : ["L80", "L99"]
}


def build_tree(file_count, groups_json):
    """ 
        tabla hash en donde la llave es una etiqueta personalizada
        y el valor es un string del nombre de la enfermedad de la piel 
    """
    
    
    stat_tree = dict()
    for group, list_range in groups_json.items():
        temp_count = 0
        for icd, count in file_count.items():
            temp_icd = icd[0:3]
            start_letter = icd[0]
            if start_letter == "L" and list_range[1] >= temp_icd and temp_icd >= list_range[0]:
                temp_count += count  
            else:
                continue
        stat_tree[group] = temp_count
    return stat_tree


def get_range_icd(tag, group):
    # Remove the letter and get de int values of the tag
    try:
        int_value = int(tag[1:3])
    except:
        raise Exception
    # get the group by the range of the groups
    for key, value in group.items():
        min_value = int(value[0][1:])
        max_value = int(value[1][1:])
        if(min_value <= int_value and int_value <= max_value):
            return key
    return

def get_icd(tag,groups_tag):
    
    for group in groups_tag.items():
        try:
            if tag.startswith("L"):
                #tag_key = group[0]
                return get_range_icd(tag, groups_tag)
        except:
            return "No_dermatologia"    
     

def add_icd_global(notas, group):

    for nota in tqdm(notas.items()):
        nota_json = nota[1]
        notas[nota[0]]["tag_icd"] = get_icd(nota_json["dx1"], group)
    
    "Método de mutación (Mutator)"
    
    return notas

def main():
    """ Leemos el archivo count.json {'L20':200} """
    dirname = os.getcwd()
    json_file = open(f'{dirname}\\graphs\\json\\count.json')
    json_count = json.load(json_file)

    """ Estadísticas de los grupos de acuerdo a la ICD """
    get_icd_tree = build_tree(json_count, groups)
    json_file.close()
    
    json_notas_file = open(f'{dirname}\\datos\\pacientes.json') 
    json_notas = json.load(json_notas_file)
    nota_0 = json_notas["nota_0"]
    
    "Creamos tag adicional de etiquetado"
    json_idc = add_icd_global(json_notas, groups)
    gen_file_json = GenFileJson()
    gen_file_json.save_json("pacientes_icd.json", json_idc)
    

    return



if __name__ == "__main__":
    main()