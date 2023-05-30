# Importar funcion de stemming
from nltk.stem import SnowballStemmer
from sklearn.metrics import pairwise
from nltk.util import ngrams

espStemmer = SnowballStemmer("spanish")

def separarTextosStemming(lista_textos):
  lista_textos_palabras = []
  for texto in lista_textos:
    lista = []
    for palabra in texto.split():
      lista.append(espStemmer.stem(palabra))
    lista_textos_palabras.append(' '.join(lista))
  return lista_textos_palabras

# Funcion para obtener la distancia (coseno) entre múltiples textos
def separarEnegrama(lista_textos_palabras, n):
  # Generar n-gramas por parrafo
  for i in range(len(lista_textos_palabras)):
    lista_textos_palabras[i] = list(ngrams(lista_textos_palabras[i].split(), n))

  # Consolidar n-grama como una sola palabra
  textos_ngramas = []
  for texto in lista_textos_palabras:
    ls = []
    for ngrama in texto:
      ls.append(" ".join(list(ngrama)))
    textos_ngramas.append(ls)
  return textos_ngramas

def distanciaEntreTextos(lista_textos, n = 1):
  # Aplicar Stemming
  lista_textos_palabras = separarTextosStemming(lista_textos)
  
  # Separar texto por enegramas
  textos_ngramas = separarEnegrama(lista_textos_palabras, n)

  # Obtener lista de todas las palabras repetidas
  palabras_total = []
  for texto in textos_ngramas:
    for palabra in texto:
      if palabra not in palabras_total:
        palabras_total.append(palabra)

  #Generar matriz
  matriz_vectores = []
  for texto in textos_ngramas:
    vector_texto = []
    for palabra in palabras_total:
      vector_texto.append(1 if palabra in texto else 0)
    matriz_vectores.append(vector_texto)
        
  # Analizar similitudes
  return pairwise.cosine_similarity(matriz_vectores)