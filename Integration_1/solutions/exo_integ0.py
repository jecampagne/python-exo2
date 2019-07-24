#Notons la subtilite que l'on doit s'assurer que n est bien un entier. 
def integ0(a,b,n,f):
    x = np.linspace(a,b,n.astype(int),endpoint=False)
    y = f(x)
    return (b-a)/n * np.sum(y)
