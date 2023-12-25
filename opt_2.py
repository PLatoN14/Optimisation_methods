import numpy as np

def func(x):
  n = len(x)
  res = 0
  for i in range(n // 2):
    res += (x[2*i - 1]**2 + x[2*i] - 11)**2 + (x[2*i - 1]**2 + x[2*i]**2 - 7)**2
  return res


def grad(x):
  n = len(x)
  res = [0] * n
  for i in range(n // 2):
    res[2*i] += 2*(x[2*i - 1]**2 + x[2*i] - 11) + 4*(x[2*i - 1]**2 + x[2*i]**2 - 7)*x[2*i]
    res[2*i - 1] += 4*(x[2*i - 1]**2 + x[2*i] - 11)*x[2*i - 1] + 4*(x[2*i - 1]**2 + x[2*i]**2 - 7)*x[2*i - 1]
  return np.array(res)


def normal(vec):
    res = 0
    for x in vec:
        res += x * x
    return res ** 0.5


def goldstein(func, grad, x_k, d, max_alpha=1, epsi=0.001, t=2):
    phi_0 = func(x_k)
    dphi_0 = np.dot(grad(x_k), d)
    a = 0
    b = max_alpha
    k = 0
    np.random.seed(42)
    alpha = np.random.rand() * max_alpha
    max_iter = 100
    while k < max_iter:
        phi = func(x_k + d * alpha)
        if phi_0 + epsi * alpha * dphi_0 >= phi:
            if phi_0 + (1 - epsi) * alpha * dphi_0 <= phi:
                break
            else:
                a = alpha
                if b >= max_alpha:
                    alpha = t * alpha
                    k += 1
                    continue
        else:
            b = alpha
        alpha = 0.5*(a + b)
        k += 1
    return alpha


def polak_ribieie(func, grad, x0, epsilon):
    k = 0
    p, x, alpha, beta = [-grad(x0)], [x0], [], ['freak']
    while normal(grad(x[k])) > epsilon:
        alpha += [goldstein(func, grad, x[k], p[k])]
        x += [x[k] + alpha[k] * p[k]]
        beta += [normal(grad(x[k + 1])).transpose() * (normal(grad(x[k + 1])) - normal(grad(x[k]))) / normal(grad(x[k])) ** 2]
        p += [-grad(x[k + 1]) + beta[k + 1] * p[k]]
        k += 1
    return x[k], func(x[k]), k


x0 = np.array([1] * 10)
point, res, num= polak_ribieie(func, grad, x0, epsilon=0.001)

print('Obtained point: %s' % (point))
print('Value of the function: %s' % (res))
print('Number of interation: %s' % (num))

