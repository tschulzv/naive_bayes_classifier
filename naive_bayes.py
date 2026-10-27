#
# Universidad Nacional de Itapua
# Inteligencia Artificial - 7mo Semestre
#
# Proyecto#3: Clasificador de Bayes
#
# Alumna: Tania Priscila Schulz Villalba
#

from os import listdir
from os.path import isfile,join
from collections import defaultdict
import re

##
#  Modelo: guarda datos para clasificar
#
class ModeloNaiveBayes:
   def __init__(self):
      self.total_documentos = 0
      self.vocabulario = set()
      self.clases = {}

# inicializa una nueva clase para el modelo
def inicializar_clase(clase):
   return {
      "nombre": clase,
      "diccionario": defaultdict(int), # diccionario palabra : conteo
      "cant_documentos": 0,
      "total_palabras": 0
   }

def tokenizar(texto): 
   # eliminar todo lo que no sea una palabra o un espacio
   limpio = re.sub(r'[^\w\s]', '', texto)
   # tokenizar en palabras
   return limpio.lower().split()

##
#  Esto entrena el modelo
#
#  modelo: el modelo actual que debes modificar o None si ninguno existe todavia
#  archivos: una lista de nombres de archivos
#  clase: la clase que corresponde a este archivo (Ej: 'comp.os.ms-windows.misc')
#  K: el factor de Laplace Smoothing
#
def entrenar(modelo, archivos, clase, K) :
   # inicializar el modelo
   if modelo is None:
      modelo = ModeloNaiveBayes()

   # si la clase no existe, inicializar
   if clase not in modelo.clases:
      modelo.clases[clase] = inicializar_clase(clase)
   
   # acceder a la clase
   c = modelo.clases[clase]

   # recorrer cada archivo
   for ruta in archivos:
      # abrir el archivo y leer el contenido
      # with cierra automaticamente el archivo 
      # **** probar con utf-8, si no con latin-1
      try:
         with open(ruta, 'r', encoding='utf-8', errors='ignore') as f:
            c.cant_documentos += 1
            modelo.total_documentos += 1
            contenido = f.read()
            # eliminar todo lo que no sea una palabra o un espacio
            limpio = re.sub(r'[^\w\s]', '', contenido)
            # tokenizar en palabras
            palabras = limpio.lower().split()

            for p in palabras:
               # aumentar el conteo de la palabra
               c.diccionario[p] += 1
               c.total_palabras += 1
               # actualizar modelo
               modelo.vocabulario.add(p)

      except (IOError, OSError):
         # manejar ???
         print("error")
   
   return modelo


##
# Esto hace la clasificacion usando el modelo
#
# modelo: el modelo a utilizar para clasificar
# archivo: el archivo que vas a clasificar
# K: el valor K
#
#
def clasificar(modelo, archivo, K) :
   try:
      with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
         contenido = f.read()
         palabras = tokenizar(contenido)
         v = len(modelo.vocabulario)
         dicc =  defaultdict(int), # diccionario palabra : conteo

         # se podria hacer reemplazando nomas cuando se encuentra uno mayor???
         #probabilidad_clases = defaultdict(int) # diccionario clase : probabilidad
         mayor_probabilidad = 0
         mayor_clase = None
   
         # aumentar el conteo de la palabra ->> ES INNECESARIO CREO??
         for p in palabras:
            dicc[p] += 1

         # probar con cada clase
         for clase in modelo.clases:
            # *** ver si hace falta aplicar K a este 
            p_clase = clase.cant_documentos / modelo.total_documentos
            p_msje_clase = 1 
            
            # recorrer cada palabra del mensaje y aplicar la formula
            for palabra in dicc: 
               p_msje_clase *= ((clase.diccionario[palabra] + K) / (clase.total_palabras + K * v))
            
            # calcular p(clase | mensaje)
            probabilidad = p_clase * p_msje_clase
            #probabilidad_clases[clase] = probabilidad
            if probabilidad > mayor_probabilidad:
               mayor_probabilidad = probabilidad
               mayor_clase = clase
   
   except (IOError, OSError):
      print("error")

   return mayor_clase
