#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Ecrire une fonction python $f(x)$ qui retourne $x(x-1)^2$

# In[ ]:


# load


# In[2]:


def f(x):
    return x*(x-1)**2


# In[47]:


x = np.linspace(0,1,50)
print(x)


# In[48]:


y = f(x)


# In[49]:


plt.plot(x,y,label=r'$f(x)$')
plt.xlabel('x',fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()


# # Calcul de l'integrale : méthode d'approximation en échellon
# $$
# \int_a^b f(x) dx = \frac{b-a}{n} \sum_{i=1}^n f(x_i) 
# $$

# In[18]:


def integ0(a,b,n,f):
    x = np.linspace(a,b,n)
    y = f(x)
    return (b-a)/n * np.sum(y)


# In[33]:


print("integrale = ",integ0(0,1,5000,f))


# In[26]:


nNodes = np.linspace(1,1000,10)
integ = []
for n in nNodes:
    integ.append(integ0(0,1,n,f))


# In[30]:


plt.plot(nNodes,np.array(integ))
plt.xscale("log")


# In[31]:


# Integrale vraie
def true_integ(a,b):
    return 1/12*(a**2*(-6 + (8 - 3*a)*a) 
                 + b**2 * (6 + b*(-8 + 3*b)))


# In[35]:


integ0_true = true_integ(0,1)
print("Integrale analytique: ",integ0_true)


# In[39]:


plt.plot(nNodes,np.array(integ)-integ0_true)
plt.xscale("log")
#plt.ylim((-0.001,0.001))


# In[42]:


nNodes = np.linspace(1000,10000,100)
integ = []
for n in nNodes:
    integ.append(integ0(0,1,n,f))


# In[43]:


plt.plot(nNodes,np.array(integ)-integ0_true)
plt.xscale("log")


# In[ ]:




