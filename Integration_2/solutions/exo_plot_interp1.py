x = np.linspace(-2,2,500)
plt.plot(x,T(1,x),label=r"$T_1(x)$")
plt.plot(x,1.5*T(1,x-1),label=r"$1.5 T_1(x-1)$")
plt.plot(x,T(1,x)+1.5*T(1,x-1),label=r"$T_1(x)+1.5T_1(x-1)$")
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.legend(fontsize=12)
plt.grid()
plt.show()
