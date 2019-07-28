def integ2(a,b,W,X,f):
    # changement d'échelle des neouds x_i
    Xnew = rescale(X,a,b)
    # calcule des valeurs y_i=f(x_i)
    Y = f(Xnew)
    # calcul de la somme pondérée
    return (b-a)/2*np.dot(Y,W)
