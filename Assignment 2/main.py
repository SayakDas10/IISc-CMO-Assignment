import numpy as np
import matplotlib.pyplot as plt
from oracles_updated import f1, f2, f3
sr_no = 24235

'''Question 1'''


def ConjugateGradient(A, b, initialx):
    '''Same algorithm that was followed in class. Because of numerical instability of 
    floating point arithmetic, instead of checking the gradient is zero, we check if it is
    'sufficiently' close to zero or not. We do this a lot in the following questions.'''

    x = initialx
    grad = np.dot(A, x) - b
    if np.all(abs(grad) < 1e-14):
        return x
    u = -grad
    i = 0
    while True:
        print(f"Iter: {i}")
        print(x)
        alpha = -np.dot(grad.T, u)/np.dot(u.T, np.dot(A, u))
        x = x + alpha * u
        grad = np.dot(A, x) - b
        if np.all(abs(grad) < 1e-14):
            return x
        beta = np.dot(u.T, np.dot(A, grad))/np.dot(u.T, np.dot(A, u))
        u = -grad + beta * u
        i += 1


def ConjugateGradientLD(A, b, initialx):
    '''Same logic as previous question. The gradient and Hessian is changed to match 
    the minimization problem of error term.'''

    x = initialx
    H = 2*np.dot(A.T, A)
    grad = np.dot(H, x) - 2*np.dot(A.T, b).T[0]
    if np.all(abs(grad) < 1e-15):
        return x
    u = -grad
    i = 0
    while True:
        print(f"Iter: {i}")
        print(x)
        alpha = -np.dot(grad.T, u)/np.dot(u.T, np.dot(H, u))
        x = x + alpha * u
        grad = np.dot(H, x) - 2*np.dot(A.T, b).T[0]
        if np.all(abs(grad) < 1e-15):
            return x
        beta = np.dot(u.T, np.dot(H, grad))/np.dot(u.T, np.dot(H, u))
        u = -grad + beta * u
        i += 1


'''Question 2'''


def plotFunctionValue(f_value):
    plt.plot(f_value)
    plt.show()


def GradientDescentQ2(alpha, initialx, iterations=100):
    '''
    Given initial alpha and x, constatnt gradient descent is calculated
    returns final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        grad = f2(x, sr_no, 1)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(f2(x, sr_no, 0))
        trajectory.append(x)
        x = x - alpha * grad

    return x, f_value, trajectory, grad_norm


def Newton(initialx, iterations=100):
    '''same algorithm that was followed in class'''
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        grad = f2(x, sr_no, 1)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(f2(x, sr_no, 0))
        trajectory.append(x)
        x = x - f2(x, sr_no, 2)

    return x, f_value, trajectory, grad_norm


'''Question 3'''


def GradientDescent(alpha, initialx, iterations=100):
    '''
    Given initial alpha and x, constatnt gradient descent is calculated
    returns final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        grad = f3(x, sr_no, 1)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(f3(x, sr_no, 0))
        trajectory.append(x)
        x = x - alpha * grad

    return x, f_value, trajectory, grad_norm


def GradientDescentDecStep(InititalAlpha, initialx, iterations=100):
    '''
    Given initial alpha and x, diminishing gradient descent is calculated
    returns final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    alpha_0 = InititalAlpha
    grad_norm = []
    f_value = []
    trajectory = []
    for i in range(iterations):
        grad = f3(x, sr_no, 1)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(f3(x, sr_no, 0))
        trajectory.append(x)
        x = x - alpha_0/(i+1) * grad

    return x, f_value, trajectory, grad_norm


def NewtonQ3(initialx, iterations=100):
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        grad = f3(x, sr_no, 1)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(f3(x, sr_no, 0))
        trajectory.append(x)
        x = x - f3(x, sr_no, 2)

    return x, f_value, trajectory, grad_norm


def NewtonsGradGame(alpha, initialx, iterations=100):
    '''We do gradient iteration for iterations-i steps followed by i newton steps.'''

    x = initialx
    x_iters = []
    fx_iters = []
    for i in range(100):
        x = initialx
        x, fx, _, _ = GradientDescent(alpha, x, iterations-i)
        x, fx_newton, _, _ = NewtonQ3(x, i)
        fx.extend(fx_newton)
        x_iters.append(x)
        fx_iters.append(fx)
    return x_iters, fx_iters


def plotForGradGame(fx, grad_iters):
    grad_fx = fx[:len(fx)-grad_iters]
    newton_fx = fx[-grad_iters:]
    plt.scatter([i for i in range(len(fx)-grad_iters)], grad_fx, marker='.')
    plt.scatter([len(grad_fx)+i+1 for i in range(grad_iters)],
                newton_fx, marker='x')
    plt.show()


'''Question 4'''


def QuasiNewtonIDApproximation(initialx, iterations=100):
    '''Same logic that is derived in the analytical solution (see pdf file).'''
    x = initialx
    I = np.identity(len(x))
    grad_norm = []
    f_value = []
    trajectory = []
    '''First iteration is outside the loop
        In the first iteration we just use id matrix as an hessian approximation
    '''
    x_prev = x
    grad_prev = f2(x, sr_no, 1)
    x = x - np.dot(I, grad_prev)
    for _ in range(iterations-1):
        grad_new = f2(x, sr_no, 1)
        gamma = grad_new - grad_prev
        delta = x - x_prev
        grad_prev = grad_new
        x_prev = x
        grad_norm.append(np.linalg.norm(grad_prev))
        f_value.append(f2(x, sr_no, 0))
        trajectory.append(x)
        # sigma = np.dot(delta.T, gamma) / np.dot(gamma.T, gamma)
        sigma = np.dot(delta.T, delta) / np.dot(delta.T, gamma)
        x = x - np.dot(sigma * I, grad_new)
        if (np.linalg.norm(grad_prev) < 1e-15):
            break

    return x, f_value, trajectory, grad_norm


def QuasiNewtonRank1(initialx, iterations=100):
    '''Same logic that was followed in class'''
    x = initialx
    I = np.identity(len(x))
    grad_norm = []
    f_value = []
    trajectory = []
    '''First iteration is outside the loop
        In the first iteration we just use id matrix as an hessian approximation
    '''
    x_prev = x
    grad_prev = f2(x, sr_no, 1)
    x = x - np.dot(I, grad_prev)
    for _ in range(iterations-1):
        grad_new = f2(x, sr_no, 1)
        gamma = grad_new - grad_prev
        delta = x - x_prev
        grad_prev = grad_new
        x_prev = x
        grad_norm.append(np.linalg.norm(grad_prev))
        f_value.append(f2(x, sr_no, 0))
        trajectory.append(x)
        sigma = delta - np.dot(I, gamma)
        I = I + np.outer(sigma, sigma)/np.dot(sigma, gamma)
        x = x - np.dot(I, grad_new)
        if (np.linalg.norm(grad_prev) < 1e-15):
            break

    return x, f_value, trajectory, grad_norm


def main():
    '''Question 1'''

    A, b = f1(sr_no, True)
    x = ConjugateGradient(A, b, np.zeros_like(b))
    print(x)
    A, b = f1(sr_no, False)
    x = ConjugateGradientLD(A, b, np.zeros_like(A[0]))
    print(x)

    '''Question 2'''

    x_0 = np.array([0, 0, 0, 0, 0])
    alpha = [0.1, 0.15, 0.2, 0.25, 0.3]
    for a in alpha:
        x, fx, _, _ = GradientDescentQ2(a, x_0, 100)
        print(x)
        plotFunctionValue(fx)

    x, f_value, _, _ = Newton(x_0, iterations=100)
    plotFunctionValue(f_value)
    print(x)

    initial_points = [np.array([5, 1, 2, 9, 0]), np.array([-1, -2, -3, -4, -5]),
                      np.array([1, 0, 6, 0, -8]
                               ), np.array([200, 683, 169, -947, -1]),
                      np.array([1900, -1683, -7169, -947, 45])]
    for x_0 in initial_points:
        x, f_value, _, _ = Newton(x_0, iterations=100)
        print(x)
        plotFunctionValue(f_value)

    '''Question 3'''

    x_0 = np.array([1, 1, 1, 1, 1])
    x, f_value, _, _ = GradientDescent(0.1, x_0)
    plotFunctionValue(f_value)
    print(min(f_value))
    print(x)

    x, f_value, _, _ = GradientDescentDecStep(0.1, x_0)
    plotFunctionValue(f_value)
    print(min(f_value))
    print(x)

    x, f_value, _, _ = NewtonQ3(x_0, iterations=10)
    print(x)
    plotFunctionValue(f_value)

    x, fx = NewtonsGradGame(0.01, x_0, 100)
    # we set the minimum of the final value of each experiment as cost
    cost = min(i[-1] for i in fx)
    optimals = []
    for i in range(len(fx)):  # check which experiments achieves that cost
        if cost == fx[i][-1]:
            optimals.append(i)
    print(cost)
    print(optimals)  # minimum of the list is the optimal cost iteration
    plotForGradGame(fx[3], 3)

    '''Question 4'''

    x_0 = np.array([0, 0, 0, 0, 0])
    x, f_value, _, _ = QuasiNewtonIDApproximation(x_0, iterations=100)
    plotFunctionValue(f_value)
    print(x)

    x, f_value, _, _ = QuasiNewtonRank1(x_0, iterations=100)
    plotFunctionValue(f_value)
    print(x)


if __name__ == "__main__":
    main()
