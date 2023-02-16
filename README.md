# Primeros ejercicios del curso de Ciencia de Datos

**First exercises for the Data Science course from the Physics MSc at Universidad de los Andes. Repository for personal study.**

Primeros ejercicios del curso de Introducción a la Ciencia de los datos.

El ejercicio 1 (ejercicio1.py) trata de hacer *fits perfectos con matriz inversa*. Contiene:
- Procedimiento para **abrir archivos** con listas de datos númericos sin NumPy
- El comando para generar una **matriz a partir de repetición de un vector** con el algoritmo para una matriz de variables polinómicas
- Una buena variedad de comandos para realizar **subplots** con la función plt.subplots(). No se ha podido asignar el título a la figura guardada.

El ejercicio 2 (nbEjercicio2.ipynb,RubioTomas_Ejercicio02.py) trata de hacer *fits aproximados con pseudoinversa*. Contiene:
- El comando para generar una **matriz a partir de repetición de un vector** con el algoritmo para una matriz de variables polinómicas.
- Un método de una línea para **evaluar polinomios**.
- El comando para calcular el **chi-cuadrado**.
- Implementación de una clase análoga a **sklearn.linear_model.LinearRegresion** para hacer regresiones fácilmente.

Los ejercicios 3 y 4 se encuentran en sus propios repositorios. Respectivamente son: reducción de parámetros con *bootstrap* y reducción de parámetros con *lasso*.

El ejercicio 5 (bayes.py) trata sobre utilizar *estadística bayesiana para encontrar un estimador de probabilidad en una distribución binomial* (lanzamientos de moneda). Contiene:
- Un método para establecer **frecuencias de caracteres** de una cadena.
- Un método para generar cadenas de caracteres aleatorios con **comprensión de listas**.
- Una función de **verosimilitud binomial**, una gaussiana implementada, y una función de **prior constante en un rango**.
- Procedimientos para hacer **derivación forward** en una línea con **slicing**.
- Un procedimiento *rudimentario* para evaluar una **probabilidad posterior** y **normalizarla**.

El ejercicio 6 (Cauchy.ipynb; Gauss.ipynb,GaussConMarkov.ipynb) es sobre *encontrar estimadores en distribuciones de Cauchy* con estadística bayesiana, y *encontrar estimadores en distribuciones con incertidumbre variada*. Contiene:
- Un procedimiento iterado para evaluar las verosimilitudes de muchos datos distintos.
- Una función de **verosimilitud de Cauchy** y una función de **prior constante en un rango** (*Cauchy.ipynb*).
- Una función de **verosimilitud Gaussiana** y una función de **prior constante en un rango** (*GaussConMarkov.ipynb*).
- Un método más sofisticado para evaluar una **probabilidad posterior** y normalizarla. Utiliza **logaritmos para estabilidad númerica** y una técnica simple para **normalizar desde el log_posterior**.
- Un procedimiento para sacar **segunda derivada forward** en una línea con **slicing**.
- Una demostración visual de la inexistencia/no-convergencia de momentos en la distribución de Cauchy.
- Una implementación de **MCMC (cadena de Markov con Metropolis-Hastings)** para generar datos aleatorios *que sigan la misma distribución de datos medidos*.
- Comandos *básicos* para realizar **histogramas**.
