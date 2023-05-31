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

from unittest import TestCase

class TestStringMethods(TestCase):

    def test_stemming(self):
      self.assertEqual(separarTextosStemming(
        ["Esta es la distancia de Coseno", "Esta otra es la distancia Euclidiana", "y esta otra es la de manhattan"]), 
        ['esta es la distanci de cosen', 'esta otra es la distanci euclidian', 'y esta otra es la de manhatt'])
      
    def test_ennegrams(self):
      self.assertEqual(separarEnegrama(
        ["Esta es la distancia de Coseno", "Esta otra es la distancia Euclidiana", "y esta otra es la de manhattan"], 2), 
        [['Esta es', 'es la', 'la distancia', 'distancia de', 'de Coseno'], ['Esta otra', 'otra es', 'es la', 'la distancia', 'distancia Euclidiana'], ['y esta', 'esta otra', 'otra es', 'es la', 'la de', 'de manhattan']])

    def test_distancia(self):
      self.assertEqual(distanciaEntreTextos(
        ["Esta es la distancia de Coseno", "Esta otra es la distancia Euclidiana", "y esta otra es la de manhattan"], 3)[0][1],
        0.25)
class distanciaCoseno(TestCase):

    def test_unigrama(self):
        self.assertEqual(0,
                         distanciaEntreTextos(["Arbol verde","Cable amarillo"])[0][1])

    def test_bigrama(self):
        self.assertEqual(0.18257418583505539,
                         distanciaEntreTextos(["Este es el texto numero uno", "Esta otro texto es el numero dos"],2)[0][1])

    def test_trigrama(self):
        self.assertEqual(0.4472135954999579,
                         distanciaEntreTextos(["Este es el texto numero uno", "Esta otro es el texto numero dos"],3)[0][1])
    
    def test_resultLength(self):
        self.assertEqual(4,
                         len(distanciaEntreTextos(["uno","dos","tres","cuatro"])))

TestStringMethods().test_stemming()
TestStringMethods().test_ennegrams()
TestStringMethods().test_distancia()
distanciaCoseno().test_unigrama()
distanciaCoseno().test_bigrama()
distanciaCoseno().test_trigrama()
distanciaCoseno().test_resultLength()
