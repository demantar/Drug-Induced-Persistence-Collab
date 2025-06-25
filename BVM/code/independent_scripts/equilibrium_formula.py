from sympy import *
# A short script to symbolically calculate the formula for the equilibrium distribution
lambda0, lambda1, mu, nu = symbols("lambda0, lambda1, mu, nu")


inf_gen = Matrix([[lambda0 - mu, mu], [nu, lambda1 - nu]])

for eig, mult, vec in inf_gen.left_eigenvects():
    print(f'eigenvalue = ${latex(eig)}$')
    print(f'normalized eigenvec = ${latex(vec[0] / sum(vec[0]))}$')
