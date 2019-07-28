def root(f,a,b,eps=1e-13):
    """ Recherche du racine de f(x)=0 sur l'intervalle (a,b).
        On donne une estimation de l'erreur absolue sur la racine
    """
    assert f(a)*f(b) <0, "Oh! f(a) et f(b) doivent être de signe opposé."
    assert eps>1e-14, "Oh! faut pas exagérer la précision :)"
    xlow = a
    xup  = b
    while np.abs(xlow-xup)>eps :
        xmiddle = (xlow+xup)/2
        if f(xlow)*f(xmiddle)>0 :
            xlow = xmiddle
        else:
            xup = xmiddle
    return (xlow+xup)/2
