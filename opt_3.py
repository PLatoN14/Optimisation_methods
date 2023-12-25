import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize


# Objective function
def f(x):
    n = len(x)
    res = 0
    val = 0
    for i in range(n - 1):
        res += (x[i]**2 - 2)**2
    for i in range(n):
        val += x[i]**2
    res += (val - 0.5) ** 2
    return res


# Gradient
def f1(x):
    n = len(x)
    res = [0] * n
    val = 0
    for i in range(n - 1):
        res[i] += 2*(x[i]**2 - 2) * 2 * x[i]
    for i in range(n):
        val += x[i] ** 2
    val -= 0.5
    for i in range(n):
        res[i] += 2 * val * 2 * x[i]
    return np.array(res)


def bfgs_method(f, f_p, x0, epsi=0.001):
    k = 0
    g_fk = f_p(x0)
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0

    while ln.norm(g_fk) > epsi:

        pk = -np.dot(Hk, g_fk)
        line_search = sp.optimize.line_search(f, f1, xk, pk)
        alpha_k = line_search[0]
        #print(line_search)

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        g_fkp1 = f_p(xkp1)
        yk = g_fkp1 - g_fk
        g_fk = g_fkp1

        k += 1

        dist = 1.0 / (np.dot(yk, sk))
        A1 = I - dist * sk[:, np.newaxis] * yk[np.newaxis, :]
        A2 = I - dist * yk[:, np.newaxis] * sk[np.newaxis, :]
        Hk = np.dot(A1, np.dot(Hk, A2)) + (dist * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    return (xk, k)


result, k = bfgs_method(f, f1, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

print('Final Result (obtained point): %s' % (result))
print('Iteration Count: %s' % (k))
print('Corresponding value of the objective function f: %s' % f(result))