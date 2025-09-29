
import numpy as np
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Valores de n para cada población
n_values = [50, 100, 200, 500]
k_values = [1, 3, 5, 11, 21]
def Mezcla(mu1,sigma1,mu2,sigma2):
    for n1, n2, k in itertools.product(n_values, n_values, k_values):
      #Se generan las normales con esos parámetros
      np.random.seed(42)
      sample1 = np.random.multivariate_normal(mu1, sigma1, n1)#muestra de no
      sample2 = np.random.multivariate_normal(mu2, sigma2, n2)#muestra de si
      X_train = np.concatenate((sample1, sample2), axis=0)
      y_train = np.concatenate((np.zeros(n1), np.ones(n2)))
      np.random.seed(44)
      sample11 = np.random.multivariate_normal(mu1, sigma1, n1)#muestra de no
      sample22 = np.random.multivariate_normal(mu2, sigma2, n2)#muestra de si
      X_train = np.concatenate((sample11, sample22), axis=0)
      y_train = np.concatenate((np.zeros(n1), np.ones(n2)))
      
      #NAIVES BAYES
      nb = GaussianNB()
      nb.fit(X_train, y_train)


      

    
    return resultado  # Opcional





# Normales muy separadas
np.random.seed(42)  # Para reproducibilidad

# Población 1: Normal centrada en (0,0)
mu1 = np.array([-3, -3])
sigma1 = np.array([[1, -0.4], -[0.4, 1]])  # Matriz de covarianza

# Población 2: Normal centrada en (3,3) - separada de la primera
mu2 = np.array([3, 3])
sigma2 = np.array([[1, -0.3], [-0.3, 1]])  # Diferente estructura de covarianza

# Generar muestras
sample1 = np.random.multivariate_normal(mu1, sigma1, n1)
sample2 = np.random.multivariate_normal(mu2, sigma2, n2)




