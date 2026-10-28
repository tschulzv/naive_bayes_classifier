#
# Universidad Nacional de Itapua
# Inteligencia Artificial - 9no Semestre
#
# Proyecto#3: Clasificador de Bayes
#
# Alumna: Tania Priscila Schulz Villalba
#

from os import listdir
from os.path import isfile,join
from collections import defaultdict
import re
import math

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
      "cant_documentos": 0,
      "total_palabras": 0,
      "diccionario": defaultdict(int) # diccionario palabra : conteo
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
            contenido = f.read()
      except (IOError, OSError) as e:
            print(f"Error al leer {ruta}: {e}")
            return None

      c["cant_documentos"] += 1
      modelo.total_documentos += 1
      # tokenizar en palabras
      palabras = tokenizar(contenido)

      for p in palabras:
         # aumentar el conteo de la palabra
         c["diccionario"][p] += 1
         c["total_palabras"] += 1
         # actualizar modelo
         modelo.vocabulario.add(p)
   
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
   if K == 0: K = 1 # para evitar logaritmo 0

   try:
      with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
         contenido = f.read()
   except (IOError, OSError) as e:
         print(f"Error al leer {archivo}: {e}")
         return None

   palabras = tokenizar(contenido)
   v = len(modelo.vocabulario)
   
   mayor_probabilidad = float('-inf')
   mayor_clase = None

   # probar con cada clase
   for nombre in modelo.clases:
      clase = modelo.clases[nombre]
      # *** ver si hace falta aplicar K a este 
      p_clase = clase["cant_documentos"] / modelo.total_documentos
      # se coonvierte a logaritmos, para evitar underflow
      log_clase = math.log(p_clase) 
            
      # recorrer cada palabra del mensaje y aplicar la formula
      # sumamos en vez de multiplicar, por propiedad de logaritmos
      for palabra in palabras:
         conteo = clase["diccionario"].get(palabra, 0)
         p_palabra = ((conteo + K) / (clase["total_palabras"] + K * v))
         log_clase += math.log(p_palabra)

         # si es la probabilidad mas alta, actualizar
         if log_clase > mayor_probabilidad:
            mayor_probabilidad = log_clase
            mayor_clase = nombre
   
   return mayor_clase
