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
    dirname = os.getcwd()
    logging.basicConfig(filename=f'{dirname}/tesis/ngrams/error_logs.txt', level=logging.ERROR)
    logging.error(e)


# Función para obtener los ngrams sintácticos de cada una de las notas clíunicas
# Ejemplo de ngrams: [['nsubj', 'advmod', 'root', 'obj', 'nummod'], ['advmod', 'root', 'obj', 'nummod', 'obl']]
def ngrams(len_ngrams, nombre_archivo, ambiente):

    try:

        dirname = os.getcwd()
        file_json = f"{dirname}/tesis/datos/pacientes_icd.json"
        data_frame = DataFrame()
        data_frame_json = data_frame.getDataframe(file_json)
        
        fechas_sort = data_frame_json.sort_values(by=["fecha"])
        
        fechas_sort_2015 = fechas_sort[fechas_sort["fecha"] >= "2015"]
        nlp = spacy.load("es_core_news_sm")

        if(ambiente == "desarrollo"):
            fechas_sort_2015 = fechas_sort_2015.head(10)
        n = len_ngrams

        ngram_notes = []
        print(f"Generando {len_ngrams}-ngrams por cada nota clínica en el dataset")
        for note in tqdm(fechas_sort_2015.values):
        
            regex_especial =  r"(S|s)[\.|\:]\s*(.+?)(Plan|P|p)[\.|\:]"

            note_original = re.sub(r"\n",".", note[6])
            resultado_regex = re.search(regex_especial, note_original)

            if resultado_regex:
                temp = resultado_regex.group(2)
                note[6] = temp
            """ else:
                print("No se encontró el patrón")
                continue """
            doc = nlp(note[6])
            #html = displacy.render(doc, style='dep', options={'compact': True})
            """ with open(f"{dirname}/tree.html", "w", encoding="utf-8") as f:
                f.write(html) """
            """ Algoritmo para generar ngrams """
            tokens = [token.dep_.lower() for token in doc if not token.is_punct and not token.is_space]
            ngrams = generate_ngrams(tokens, n)
            ngram_notes.append([ngrams, note[13]])
        print("Guardando ngrams en un archivo csv para su posterior uso")
        print(f"{ngram_notes.__len__()} notas generadas")
        df_ngrams = pd.DataFrame(ngram_notes,
                                columns=["ngrams","tag_icd"])
        save_file = f"{dirname}/tesis/ngrams/{ambiente}_{nombre_archivo}"
        df_ngrams.to_pickle(save_file)

    except Exception as e:
        write_error_logs(e)
    return




# define a function to generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]



    
def dictNgrams(n, nombre_archivo_csv,nombre_archivo,ambiente):
    try:
        dirname = os.getcwd()
        file_name = f"{dirname}/tesis/ngrams/{ambiente}_{nombre_archivo_csv}"
        dataframe_ngrams = pd.read_pickle(file_name)["ngrams"]
        
        if(ambiente == "desarrollo"):
            dataframe_ngrams = dataframe_ngrams.head(10)
        print(f"Generando diccionario de ngrams {n} diferentes")
        dict_ngrams = {str(ngrams) for notes in tqdm(dataframe_ngrams.values) for eval_notes in [notes] for ngrams in eval_notes}
        
        print(f"{len(dict_ngrams)} {n}-ngrams diferentes generados")

        with open(f"{dirname}/tesis/ngrams/dict_{ambiente}_{nombre_archivo}", "w") as archivo:
            pass
        with open(f"{dirname}/tesis/ngrams/dict_{ambiente}_{nombre_archivo}", "wb",) as file:
            pk.dump(dict_ngrams, file)

    except Exception as e:
        write_error_logs(e)
    return

# Bag of words
def countNgrams(file, file_csv,nombre_archivo_matrix,ambiente):
    try:
        dirname = os.getcwd()
        dict_file = f"{dirname}/tesis/ngrams/{ambiente}_{file_csv}"
        dataframe_ngrams = pd.read_pickle(dict_file)["ngrams"]
        dataframe_ngrams_tags =pd.read_pickle(dict_file)["tag_icd"]
        dict_ngrams_file = f"{dirname}/tesis/ngrams/dict_{ambiente}_{file}"
        with open(f"{dict_ngrams_file}", "rb") as archivo:
            set_file = pk.load(archivo)
            data_aray = list(set_file)
            data_aray = [eval(ngram) for ngram in data_aray]
            series_data = pd.Series(data_aray).T
        columnas = int(series_data.shape[0])
        filas = int(dataframe_ngrams.shape[0])
        # matriz para guardar la cantidad de ngrams en cada nota clínica
        matrix_ngrams = np.zeros((filas, columnas))
        print("Contando ngrams, último paso")
        # ciclo para recorrer las notas clínicas y contar los ngrams

        for i, row_notas in tqdm(enumerate(dataframe_ngrams), total=len(dataframe_ngrams)):
            # lista de ngrams en la nota clínica
            #list_row = eval(row)
            for j, dict_ngrams in enumerate(series_data):
                #ngram_dict = eval(dict_ngrams)
                if dict_ngrams in row_notas:
                    matrix_ngrams[i][j] += 1
        # se crea un dataframe con la matriz de ngrams; resultado del conteo de ngrams en cada nota clínica.
        # Cada columna es el conteo de ngram (bag of words) y cada fila es una nota clínica.
        df_ngrams = pd.DataFrame(matrix_ngrams)
        df_ngrams = pd.concat([df_ngrams,dataframe_ngrams_tags], axis=1)
        matrix_file = f"{dirname}/tesis/ngrams/{ambiente}_{nombre_archivo_matrix}"
        df_ngrams.to_pickle(matrix_file)
        print(f"Matriz de ngrams guardada en {matrix_file}, algoritmo termino con éxito")
        return
    except Exception as e:
        write_error_logs(e)



def dict_count_tokens(file, file_pkl,nombre_archivo_matrix,ambiente):
    try:
        dirname = os.getcwd()
        dict_file = f"{dirname}/tesis/ngrams/{ambiente}_{file_pkl}"
        dataframe_ngrams = pd.read_pickle(dict_file)["ngrams"]
        dataframe_ngrams_tags =pd.read_pickle(dict_file)["tag_icd"]
        dict_ngrams_file = f"{dirname}/tesis/ngrams/dict_{ambiente}_{file}"
        with open(f"{dict_ngrams_file}", "rb") as archivo:
            set_file = pk.load(archivo)
            data_aray = list(set_file)
            data_aray = [eval(ngram) for ngram in data_aray]
            series_data = pd.Series(data_aray).T
        columnas = int(series_data.shape[0])
        filas = int(dataframe_ngrams.shape[0])
        # matriz para guardar la cantidad de ngrams en cada nota clínica
        matrix_ngrams = np.zeros((filas, columnas))
        print("Contando TF de tokens, último paso")
        # ciclo para recorrer las notas clínicas y contar los ngrams

        dict_token_count = dict()

        for dict_ngrams in tqdm(series_data):
            count = 0
            for row_notas in dataframe_ngrams:
                if dict_ngrams in row_notas:
                    count += 1
                    dict_string_token = str(dict_ngrams) 
                    dict_token_count[dict_string_token] = count
                else:
                    dict_string_token = str(dict_ngrams)
                    dict_token_count[dict_string_token] = count
        print(f'{len(dict_token_count)} tokens diferentes generados con su respectivo conteo TF')
        dict_tf_file = f"{dirname}/tesis/ngrams/dict_tf_{ambiente}_{file}"
        with open(dict_tf_file, "w") as archivo:
            pass
        with open(dict_tf_file, "wb",) as file:
            pk.dump(dict_token_count, file)
    except Exception as e:
        write_error_logs(e)
    return

def count_tf_ngrams(file, file_csv,nombre_archivo_matrix,ambiente):
    
    try:
        dirname = os.getcwd()
        dict_file = f"{dirname}/tesis/ngrams/{ambiente}_{file_csv}"
        dataframe_ngrams = pd.read_pickle(dict_file)["ngrams"]
        dataframe_ngrams_tags =pd.read_pickle(dict_file)["tag_icd"]
        dict_ngrams_file = f"{dirname}/tesis/ngrams/dict_tf_{ambiente}_{file}"
        with open(f"{dict_ngrams_file}", "rb") as archivo:
            set_file = pk.load(archivo)
            data_aray = list(set_file)
            data_aray = [eval(ngram) for ngram in data_aray]
            series_data = pd.Series(data_aray).T
        columnas = int(series_data.shape[0])
        filas = int(dataframe_ngrams.shape[0])
        # matriz para guardar la cantidad de ngrams en cada nota clínica
        matrix_ngrams = np.zeros((filas, columnas))
        print("Contando ngrams, último paso")
        # ciclo para recorrer las notas clínicas y contar los ngrams

        for i, row_notas in tqdm(enumerate(dataframe_ngrams), total=len(dataframe_ngrams)):
            # lista de ngrams en la nota clínica
            #list_row = eval(row)
            for j, dict_ngrams in enumerate(series_data):
                #ngram_dict = eval(dict_ngrams)
                if dict_ngrams in row_notas:
                    temp= set_file[str(dict_ngrams)]
                    matrix_ngrams[i][j] = temp
        # se crea un dataframe con la matriz de ngrams; resultado del conteo de ngrams en cada nota clínica.
        # Cada columna es el conteo de ngram (bag of words) y cada fila es una nota clínica.
        df_ngrams = pd.DataFrame(matrix_ngrams)
        df_ngrams = pd.concat([df_ngrams,dataframe_ngrams_tags], axis=1)
        matrix_file = f"{dirname}/tesis/ngrams/tf_{ambiente}_{nombre_archivo_matrix}"
        df_ngrams.to_pickle(matrix_file)
        print(f"Matriz de ngrams guardada en {matrix_file}, algoritmo termino con éxito")
        return
    except Exception as e:
        write_error_logs(e)


if __name__ == "__main__":
    n = 2
    nombre_archivo = f"dictionary_ngrams{n}.pkl"
    nombre_archivo_csv = f"dictionary_ngrams{n}_prueba.pkl"
    nombre_archivo_matrix = f"dictionary_ngrams{n}_matrix.pkl"
    ambiente = "produccion"
    nombre_archivo_tf = f"dictionary_ngrams{n}.pkl"
    
    ngrams(n, nombre_archivo_csv, ambiente)
    dictNgrams(n, nombre_archivo_csv,nombre_archivo,ambiente)
    #countNgrams(nombre_archivo,nombre_archivo_csv,nombre_archivo_matrix,ambiente)
    dict_count_tokens(nombre_archivo,nombre_archivo_csv,nombre_archivo_matrix,ambiente)
    count_tf_ngrams(nombre_archivo,nombre_archivo_csv,nombre_archivo_matrix,ambiente)