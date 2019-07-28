def legendre(n,x):
    """ Calcul de P_n(x) et P_{n-1}(x)
        par récurrence.
    """
    assert n>=0, "Oh! n doit être >=0"
    n = int(n) # juste pour être sure que n est un entier    
    if n==0:
        return {'pn': 1, 'pnm1': 0}
    elif n==1:
        return {'pn': x, 'pnm1':1}
    else:
        p0=1
        p1=x
        for i in range(2,n+1):
            p2 = ((2*i-1)*x*p1-(i-1)*p0)/i
            p0 = p1
            p1 = p2
        return {'pn': p1, 'pnm1':p0}
