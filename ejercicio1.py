import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


file = open('ejercicios/01/numeros_20.txt')
x = []
y = []
for line in file.readlines():
    nums = line.split()
    x.append(float(nums[0]))
    y.append(float(nums[1]))
x = np.array(x)
y = np.array(y)


def regresion(x,y):
    S = np.power(np.transpose(np.repeat(x[None],len(x),axis=0)),np.linspace(0,len(x)-1,len(x)))
    Sinv = la.inv(S)
    return np.dot(Sinv,y)

def poly(coef,x):
    powers = np.power(np.transpose(np.repeat(x[None],len(coef),axis=0)),np.linspace(0,len(coef)-1,len(coef)))
    return np.sum(np.multiply(powers,coef),axis=1)



fig,ax = plt.subplots(2,2,gridspec_kw={'hspace':0.5,'wspace':0.35})
axes = list(ax[0]) + list(ax[1])
fig.suptitle('Regresiones exactas')

for i in range(2,6):
    x_prime = x[:i]
    y_prime = y[:i]
    coefs = regresion(x_prime,y_prime)
    x_test = np.linspace(0,1)
    
    a = axes[i-2]
    a.scatter(x_prime,y_prime)
    a.plot(x_test,poly(coefs,x_test))
    a.set_title('M = '+str(i-1))

for ax in axes:
    ax.set(xlabel='x',ylabel='y')
    ax.label_outer()

fig.set_size_inches(8,5)
#fig.tight_layout()

fig.savefig('Regresiones.png')