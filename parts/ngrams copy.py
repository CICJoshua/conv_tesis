""" 
df_ngrams

This function creates ngrams from a json file. It first gets the current working directory and the file path of the json file. Then it creates a dataframe from the json file and sorts it by date. It then filters out any dates before 2015. 

The function then uses spacy to tokenize the notes, generate ngrams, and render a dependency tree. Finally, it stores the ngrams in a csv file and returns a dataframe with the ngrams and tag_icd columns.

"""




import os
from dataFrame import DataFrame
import re
import spacy
from spacy import displacy
from tqdm import tqdm
import pandas as pd
import pickle as pk
import numpy as np
import logging
def write_error_logs(e):
    logging.basicConfig(filename='/001/usuarios/joshuaguerrero/tesis/parts/error_logs.txt', level=logging.ERROR)
    logging.error(e)

def ngrams(len_ngrams, nombre_archivo, ambiente):


    dirname = os.getcwd()
    file_json = f"{dirname}/datos/pacientes_icd.json"
    data_frame = DataFrame()
    data_frame_json = data_frame.getDataframe(file_json)
    
    fechas_sort = data_frame_json.sort_values(by=["fecha"])
    
    fechas_sort_2015 = fechas_sort[fechas_sort["fecha"] >= "2015"]
    nlp = spacy.load("es_core_news_sm")

    if(ambiente == "desarrollo"):
        fechas_sort_2015 = fechas_sort_2015.head(10)
    n = len_ngrams

    ngram_notes = []
    print("Generando ngrams")
    for note in tqdm(fechas_sort_2015.values):
      
        regex_especial =  r"(S|s)[\.|\:]\s*(.+?)(Plan|P|p)[\.|\:]"
        """ regex_especial_mayusculas =  r"S\.|\:\s*(.+?)\s*P\.|\:"
        resultado_regex_minusculas = re.search(regex_especial_minusculas, note[6])
        resultado_regex_mayusculas = re.search(regex_especial_mayusculas, note[6])

        if resultado_regex_minusculas:
            temp = resultado_regex_minusculas.string
            note[6] = temp
        elif resultado_regex_mayusculas:
            temp = resultado_regex_mayusculas.string
            note[6] = temp
        else:
            continue """
        note_original = re.sub(r"\n",".", note[6])
        resultado_regex = re.search(regex_especial, note_original)

        if resultado_regex:
            temp = resultado_regex.group(2)
            note[6] = temp
        else:
            continue
        doc = nlp(note[6])
        html = displacy.render(doc, style='dep', options={'compact': True})
        """ with open(f"{dirname}/tree.html", "w", encoding="utf-8") as f:
            f.write(html) """
        """ Algoritmo para generar ngrams """
        tokens = [token.dep_.lower() for token in doc if not token.is_punct and not token.is_space]
        ngrams = generate_ngrams(tokens, n)
        ngram_notes.append([ngrams, note[13]])

        """ Algoritmo para generar ngrams sintácticos """
        """ for sent in doc.sents:
            root = sent.root
            print(root.text, root.dep_, root.head.text)
            # imprimir los hijos de la raíz y sus dependencias

            for child in root.children:
                print(child.text, child.dep_, child.head.text)
                # imprimir los hijos de los hijos y sus dependencias
                for subchild in child.children:
                    print(subchild.text, subchild.dep_, subchild.head.text)
            break """

    df_ngrams = pd.DataFrame(ngram_notes,
                             columns=["ngrams","tag_icd"])
    

    
    
    df_ngrams.to_csv(f"{dirname}/{nombre_archivo}")
    return




# define a function to generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]



    
def dictNgrams(n, nombre_archivo_csv,nombre_archivo,ambiente):
    dirname = os.getcwd()
    file_name = f"{dirname}/{nombre_archivo_csv}"
    dataframe_ngrams = pd.read_csv(file_name)
    
    if(ambiente == "desarrollo"):
        dataframe_ngrams = dataframe_ngrams.head(50)
    print("Generando diccionario de ngrams")
    dict_ngrams = {str(ngrams) for notes in tqdm(dataframe_ngrams.values) for eval_notes in [eval(notes[1])] for ngrams in eval_notes}
    
    print(len(dict_ngrams))

    with open(f"{dirname}/dict_{nombre_archivo}", "w") as archivo:
        pass
    with open(f"{dirname}/dict_{nombre_archivo}", "wb",) as file:
        pk.dump(dict_ngrams, file)

    return

def countNgrams(file, file_csv,nombre_archivo_matrix):
    try:
        dirname = os.getcwd()
        dataframe_ngrams = pd.read_csv(f"{dirname}/{file_csv}")["ngrams"]
        dataframe_ngrams_tags =pd.read_csv(f"{dirname}/{file_csv}")["tag_icd"]
        with open(f"{dirname}/dict_{file}", "rb") as archivo:
            set_file = pk.load(archivo)
            data_aray = list(set_file)
            series_data = pd.Series(data_aray).T
        columnas = int(series_data.shape[0])
        filas = int(dataframe_ngrams.shape[0])

        matrix_ngrams = np.zeros((filas,columnas))
        j = 0
        print("Contando ngrams, último paso")
        for attributes in tqdm(range(len(series_data))):
            dict_ngrams = eval(series_data[attributes])
            i = 0
            for row in dataframe_ngrams:
                list_row = eval(row)
                if dict_ngrams in list_row:
                    if i == 0 and j == 0:
                        matrix_ngrams[i][j] += 1
                    elif i == filas:
                        matrix_ngrams[i-1][j] += 1
                    elif i==filas and j == columnas:
                        matrix_ngrams[i-1][j-1] += 1
                    else:
                        matrix_ngrams[i][j] += 1

                i += 1
            j += 1
        df_ngrams = pd.DataFrame(matrix_ngrams)
        df_ngrams = pd.concat([df_ngrams,dataframe_ngrams_tags], axis=1)
        df_ngrams.to_csv(f"{dirname}/{nombre_archivo_matrix}")
        return
    except Exception as e:
        write_error_logs(e)
    

if __name__ == "__main__":
    n = 10
    nombre_archivo = f"dictionary_ngrams{n}.pkl"
    nombre_archivo_csv = f"dictionary_ngrams{n}.csv"
    nombre_archivo_matrix = f"dictionary_ngrams{n}_matrix.csv"
    ambiente = "produccion"
    
    ngrams(n, nombre_archivo_csv, ambiente)
    dictNgrams(n, nombre_archivo_csv,nombre_archivo,ambiente)
    countNgrams(nombre_archivo,nombre_archivo_csv,nombre_archivo_matrix)