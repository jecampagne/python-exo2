#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Ecris une fonction python nommée $f(x)$ qui retourne la valeur $x(x-1)^2$

# In[2]:


# %load solutions/exo_func.py


# Python permet l'appel de la fonction $f(x)$ avec un tableau de valeurs "$x[0],\dots,x[N-1]$".
# Crées un tableau de 50 valeurs de $[0,1[$ , en utilisant la fonction *linspace* de la librairie **numpy**. Remarque que *linspace* a un argument qui permet d'inclure ou non les bornes; la valeur par défaut
# est "endpoint=True", or il nous faudra exclure la borne supérieure de l'intervale donc
# il faut utiliser "endpoint=False".

# In[5]:


#obtenir des informations sur une fonction decommente la ligne ci-dessous
#?np.linspace


# In[6]:


# %load solutions/exo_array.py


# In[7]:


print("taille = ",len(x)," valeurs= ",x)


# Obtenir le tableau des valeurs $f(x[0]),\dots,f(x[N-1])$ s'obtient alors en donnant le tableau $x$ à $f$ et à l'affecter à la variable $y$ qui devient également un tableau de $N$ valeurs.

# In[8]:


y = f(x)


# Le graphe de $f(x)$ s'obtient en utilisant la librairie **matplotlib**. Ici le code permet
# * de placer des couples $(x[i], f(x[i]))$ avec un marker rond bleu
# * de dessiner une courbe passant ces points
# * de donner un label à l'axe des abscisses et un autre à l'axe des ordonnées
# * de faire figurer une légende
# * de dessiner un cadrillage

# In[9]:


plt.scatter(x,y,label=r'$f(x)$')
plt.plot(x,y,c='r',ls='--')
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()


# # Calcul de l'integrale : méthode d'approximation en escalier.
# On veut donc calculer une approximation de l'intégrale selon la formule
# $$
# \int_a^b f(x) dx \approx \frac{b-a}{n} \sum_{i=0}^{n-1} f(x_i) 
# $$

# Ecris donc une fonction nommée "integ0" qui a pour arguments: 
# * la borne inférieure de l'intégrale $a$
# * la borne supérieure de l'intégrale $b$
# * le nombre de points $n$ qui sont utilisés pour calculer $f(x_i)$
# * la fonction $f$ à intégrer
# Cette fonction "integ0" doit:
# 1. créer un tableau $x$ de $n$ valeurs dans l'intervalle $[a, b[$
# 2. créer le tableau des $n$ valeurs $f(x_i)$ et le stocker dans la variable $y$
# 3. calculer la somme $\sum_i f(x_i)$, on pourra se servir de la fonction *sum* de **numpy**
# 4. retourner la valeur de l'intégrale

# In[38]:


# %load solutions/exo_integ0.py


# On peut donc maintenant facilement calculer l'approximation de l'intégrale sur $[0, 1[$ de la fonction $f(x)=x(x-1)^2$ avec $n=100$ échantillons $x_i$. Tu peux également changer la valeur de $n$ pour voir ce que cela change.

# In[28]:


n=100
print("Approx. integrale [",n,"] = ",integ0(0,1,n,f))


# Or, $f(x)=x(x-1)^2$ est un polynôme du 3éme degré dont la primitive $F(x)$ est facilement calculable à la main pour obtenir:
# $$
# \int_a^b f(x) dx = F(b)-F(a)
# $$
# Ecris donc une fonction nommée $F(x)$ qui retourne la valeur de la primitive en $x$.

# In[12]:


# %load solutions/exo_primitive.py


# In[13]:


integ0_true = F(1)-F(0)
print("'Vraie' integrale =", integ0_true)


# On note une différence entre la valeur approchée de l'intégrale obtenue avec $n=100$ échantillons entre $[0,1[$ et la valeur obtenue par le culcul directe de la primitive. Nous allons étudier l'évolution de l'approximation en fonction de $n$.
# 
# Le code ci-dessous effectue les étapes suivantes:
# 1. construction d'un tableau "nNodes" de 100 valeurs entre $[10,10^4]$ échantillonné en échelle logarithmique
# 2. on contruit une liste "integ" des intégrales approchées en les calculant par une boucle sur $n$ 
# 3. on transforme la liste en tableau **numpy**

# In[52]:


nNodes = np.logspace(1,4,100,dtype=int) 
integ = []
for n in nNodes:
    integ.append(integ0(0,1,n,f))
integ = np.array(integ)


# Le graphe suivant montre le pourcentage d'erreur commise en fonction du nombre d'échantillons $n$ qui servent à calculer l'intégrale approchée.

# In[45]:


plt.plot(nNodes,100*((integ/integ0_true)-1))
plt.xscale("log")
plt.xlabel('n',fontsize=20)
plt.ylabel(r"$(I_{approx}-I_{vraie})/I_{vraie}$ (%)",fontsize=20)
plt.grid()
plt.show()


# Plus donc $n$ augmente plus l'accord "approximation" versus "vraie" est meilleure. On conçoit en effet que plus le nombre d'échantillons $x_i$ sur $[0,1[$ augmente meilleure est l'approximation en escalier de la fonction $f(x)$. On peut présenter le résultat en prenant la valeur absolue de la différence "approx"-"vraie" et en mettant en échelle log l'axe des ordonnées.

# In[51]:


plt.plot(nNodes,100*np.abs(((integ/integ0_true)-1)))
plt.xscale("log")
plt.yscale("log",nonposy='clip')
plt.xlabel('n',fontsize=20)
plt.ylabel(r"$\|\Delta I\|/I_{vraie}$ (%)",fontsize=20)
plt.grid()
plt.show()


# En lisant ce graphe d'aprés toi, quelle est la loi d'échelle, c'est-à-dire donne la valeur de $p$ de la relation suivante: 
# $$
# \frac{|\Delta I|}{I} = \frac{1}{n^p}
# $$

# In[ ]:




