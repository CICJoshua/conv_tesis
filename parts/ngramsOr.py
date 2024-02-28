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



def ngrams(len_ngrams, nombre_archivo, ambiente):


    dirname = os.getcwd()
    file_json = f"{dirname}\\datos\\pacientes_icd.json"

    data_frame = DataFrame()
    data_frame_json = data_frame.getDataframe(file_json)


    

    fechas_sort = data_frame_json.sort_values(by=["fecha"])
    
    fechas_sort_2015 = fechas_sort[fechas_sort["fecha"] >= "2015"]
    nlp = spacy.load("es_core_news_sm")

    if(ambiente == "desarrollo"):
        fechas_sort_2015 = fechas_sort_2015.head(5)
    n = len_ngrams

    ngram_notes = []
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
        with open("D:\\tesis\\parts\\tree.html", "w", encoding="utf-8") as f:
            f.write(html)
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
    
    """ with open(f"D:\\tesis\\parts\\{nombre_archivo}", "w") as archivo:
        pass
    with open(f"D:\\tesis\\parts\\{nombre_archivo}", "wb",) as file:
        pk.dump(df_ngrams, file) """
    
    

    return




# define a function to generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return [tokens[i:i+n] for i in range(len(tokens)-n+1)]



    
def dictNgrams(n, nombre_archivo,ambiente):
    dirname = os.getcwd()
    file_name = f"{dirname}\\parts\\{nombre_archivo}"
    with open(file_name, "rb") as archivo:
        dataframe_ngrams = pd.DataFrame(pk.load(archivo))

    if(ambiente == "desarrollo"):
        dataframe_ngrams = dataframe_ngrams.head(5)
    dict_ngrams = set()
    for notes in tqdm(dataframe_ngrams.values):
        for ngrams in notes[0]:
            temp = str(ngrams)
            dict_ngrams.add(temp)
    
    print(dict_ngrams.__len__())

    with open(f"D:\\tesis\\parts\\dict_{nombre_archivo}", "w") as archivo:
        pass
    with open(f"D:\\tesis\\parts\\dict_{nombre_archivo}", "wb",) as file:
        pk.dump(dict_ngrams, file)

    return

def countNgrams(file):
    dirname = os.getcwd()
    with open(f"{dirname}\\parts\\{file}", "rb") as archivo:
        set_file = pk.load(archivo)
        dataframe_ngrams = pd.DataFrame(set_file, columns=["ngrams","tag_icd"])
    return

if __name__ == "__main__":
    n = 5
    nombre_archivo = f"dictionary_ngrams{n}.pkl"
    ambiente = "desarrollo"
    #ngrams(n, nombre_archivo, ambiente)
    #dictNgrams(n, nombre_archivo,ambiente)
    countNgrams(nombre_archivo)