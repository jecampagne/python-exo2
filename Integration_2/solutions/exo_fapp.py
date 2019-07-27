def fapp(x, xs,ys,w=1):
    ns = xs.size-1
    res = 0
    for i in range(ns+1):
        res += ys[i]*T(ns,x-xs[i],w)
    return res
