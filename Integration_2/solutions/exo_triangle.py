def Triangle(a,b,x):
    assert a<b, "Oh non! a doit �tre inf�rieur � b"
    assert a!=b, "Oh non! a doit �tre diff�rent de b"
    res = 0.;
    if x < a :
        res = 0.
    elif x < (a+b)/2:
        res = 2*(x-a)/(b-a)
    elif x < b:
        res = 2*(b-x)/(b-a)
    else:
        res = 0
    return res
