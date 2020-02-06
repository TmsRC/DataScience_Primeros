import numpy as np
import matplotlib.pyplot as plt


def frecuencias(cadena):
    lista = np.array(list(cadena))
    caras = np.sum(lista=='c')
    sellos = len(lista) - caras
    return caras,sellos

def prior(H):
    return 1*(H<=1)*(H>=0)

def verosimilitud(H,cadena):
    c,s = frecuencias(cadena)
    return H**(c)*(1-H)**(s)

def gaussiana(x,mu,sigma):
    norm = 1/(2*np.pi*sigma**2)**(1/2)
    return np.exp(-(x-mu)**2/(2*sigma**2))*norm


random = np.random.randint(0,3,np.random.randint(5,50))
cad = "".join('s' if num==0 else 'c' for num in random)
print(cad)

#cad = 'scccc'

H = np.linspace(10**-5,1,1000)
V = verosimilitud(H,cad)
Evidencia = np.trapz(V,H)
posterior = V/Evidencia

L = np.log(posterior)
der1 = (L[1:]-L[:-1])/(1/1000)
der2 = (der1[1:]-der1[:-1])/(1/1000)

cero = np.min(np.abs(der1))
print('Derivada en la raiz: ',cero)
puntoC = np.nonzero(np.abs(der1)==cero)
#print(H[puntoC])
sigma = float((-der2[puntoC])**(-1/2))
mu = float(H[puntoC])
print('Media: ',mu)
print('Desviación: ',sigma)


print('Normalizada?','Integral =', np.trapz(posterior,H))
plt.figure()
plt.plot(H,posterior,label='Real')
plt.plot(H,gaussiana(H,mu,sigma),linestyle='-.',label='Aprox. Gaussiana')
plt.xlabel('H')
plt.ylabel('Posterior')
plt.title('H = {:.2f}  $\pm$ {:.2f}'.format(mu,sigma))
plt.legend()
plt.savefig('coins.png')