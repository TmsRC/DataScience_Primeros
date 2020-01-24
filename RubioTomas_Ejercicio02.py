import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

datosFull = np.loadtxt('numeros_20.txt')
x = datosFull[:,0]
y = datosFull[:,1]

# Primera Parte ---------------------------------------------------------------------------------------------------------

def fitCoef(x,y,M):
    S = np.power(np.transpose(np.repeat(x[None],M+1,axis=0)),np.linspace(0,M,M+1))
    Sinv = la.pinv(S) #Si se puede, usar pseudoinversa implementada en linalg
    return np.dot(Sinv,y)
def polyGen(coef,x):
    M = len(coef)-1
    powers = np.power(np.transpose(np.repeat(x[None],M+1,axis=0)),np.linspace(0,M,M+1))
    return np.sum(np.multiply(powers,coef),axis=1)
def chiSquare(y,t):
    return np.sum(np.power(np.subtract(y,t),2))      

fig,axii = plt.subplots(2,2)

axes = list(axii[0])+list(axii[1])
Ms = [0,1,3,9]
E_rms = [[],[]]
for i in range(0,10):
    
    x_test = np.linspace(0,1)
    coefs = fitCoef(x[:10],y[:10],i)
    y_fit = polyGen(coefs,x_test) #Si era el otro fit, cambiar esto.
    
    E_rms[0].append(np.sqrt(chiSquare(polyGen(coefs,x[:10]),y[:10]))/10)
    E_rms[1].append(np.sqrt(chiSquare(polyGen(coefs,x[10:]),y[10:]))/10)
    
    if i in Ms:
        a = axes[Ms.index(i)]
        a.scatter(x[10:],y[10:],facecolors='none',edgecolors='b')
        a.scatter(x[:10],y[:10],c='grey')
        a.plot(x_test,y_fit,c='r')
        a.set_title("M = "+str(i))
        a.set_ylim(0,2)
        if i == 9:
            a.set_ylim(-2000,2000)
    
fig.tight_layout()
fig.savefig('PrimeraParte_Figura1.png')

Ms = range(0,10)

fig = plt.figure()
plt.plot(Ms,E_rms[0],c='r',label='Datos entrenamiento')
plt.scatter(Ms,E_rms[0],c='r')
plt.plot(Ms,E_rms[1],c='b',label='Datos prueba')
plt.scatter(Ms,E_rms[1],c='b')
plt.yscale('log')
plt.legend()
plt.savefig('PrimeraParte_Figura2.png')


# Segunda Parte ---------------------------------------------------------------------------------------------------------------

class PolyFit():
    def __init__(self,degree=0):
        self.M = degree+1
        self.betas = np.array([])
    def fit(self,x,y):
        S = np.power(np.transpose(np.repeat(x[None],self.M,axis=0)),np.linspace(0,self.M-1,self.M))
        Sinv = la.pinv(S)
        self.betas = np.dot(Sinv,y)
    def predict(self,x):
        powers = np.power(np.transpose(np.repeat(x[None],self.M,axis=0)),np.linspace(0,self.M-1,self.M))
        return np.sum(np.multiply(powers,self.betas),axis=1)
    def score(self,x,y):
        chi = np.sum(np.power(np.subtract(self.predict(x),y),2))
        return np.sqrt(chi/len(y))

fig,axii = plt.subplots(2,2)

axes = list(axii[0])+list(axii[1])
Ms = [0,1,3,9]
E_rms = [[],[]]
for i in range(0,10):
    hola = PolyFit(i)
    x_test = np.linspace(0,1)
    coefs = hola.fit(x[:10],y[:10])
    y_fit = hola.predict(x_test)
    
    E_rms[0].append(hola.score(x[:10],y[:10]))
    E_rms[1].append(hola.score(x[10:],y[10:]))
    
    if i in Ms:
        a = axes[Ms.index(i)]
        a.scatter(x[10:],y[10:],facecolors='none',edgecolors='b')
        a.scatter(x[:10],y[:10],c='grey')
        a.plot(x_test,y_fit,c='r')
        a.set_title("M = "+str(i))
        a.set_ylim(0,2)
        if i == 9:
            a.set_ylim(-2000,2000)
    
fig.tight_layout()
fig.savefig('SegundaParte_Figura1.png')


Ms = range(0,10)

fig = plt.figure()
plt.plot(Ms,E_rms[0],c='r',label='Datos entrenamiento')
plt.scatter(Ms,E_rms[0],c='r')
plt.plot(Ms,E_rms[1],c='b',label='Datos prueba')
plt.scatter(Ms,E_rms[1],c='b')
plt.yscale('log')
plt.legend()
plt.savefig('SegundaParte_Figura2.png')