from CMO_A1 import f1, f2, f3, f4 # type: ignore
import numpy as np
import matplotlib.pyplot as plt

global sr_number
sr_number = 24235

'''
    Question 1
'''


def findMinimas(fn):
    ''' Given a funciton f, calculates the minimas of f. 
    
        Returns: List of x such that f(x) is minimum
    '''
    x = np.linspace(-2, 2, 1000)
    fx = [fn(sr_number, i) for i in x]
    minima = [[x[0], fx[0]]]
    for i in range(1, len(fx)):
        if (minima[-1][1] > fx[i]):
            minima = [[x[i], fx[i]]]
        elif minima[-1][1] == fx[i]:
            minima.append([x[i], fx[i]])
    return minima


def isConvex(fn, interval):
    '''
    Given a function f and an interval, checks if the function is convex or not in the interval using Jensens Inequality
    
    Returns: True if the function is convex. False otherwise
    '''
    X = np.arange(interval[0], interval[1], 500)
    alpha = np.linspace(0, 1, 10)
    for x in X:
        for y in np.linspace(x, interval[1], 500):
            for a in alpha:
                lhs = round(fn(sr_number, a*x+(1-a)*y), 6)
                rhs = round(a*fn(sr_number, x)+(1-a)*fn(sr_number, y), 6)
                if (lhs > rhs):
                    return False
    return True


def isStrictlyConvex(fn, interval):
    '''
    Given a function f and an interval, checks if the function is strictly convex or not in the interval using Jensens strong Inequality
    
    Returns: True if the function is strictly convex. False otherwise
    '''
    if isConvex(fn, interval):
        X = np.linspace(interval[0], interval[1], 500)
        alpha = np.linspace(0, 1, 10)
        for x in X:
            for y in np.linspace(x+1e-3, interval[1], 500):
                for a in alpha:
                    lhs = round(fn(sr_number, a*x+(1-a)*y), 6)
                    rhs = round(a*fn(sr_number, x)+(1-a)*fn(sr_number, y), 6)
                    if (lhs-1e-8 >= rhs):
                        print(f"x: {x}, y: {y}, a: {a}")
                        return False
        return True
    else:
        return False


def isCoercive(fn):
    '''
    Given a funciton f, check if the function is coercive or not by discrete sampling of x such that f(x[i]) < f(x[i+1])
    returns True if the function is coercive, False otherwise
    '''
    x_positive = np.arange(10, 10000, 10)
    x_negative = -1*x_positive
    y_positive = fn(sr_number, x_positive)
    y_negative = fn(sr_number, x_negative)
    for i in range(1, len(y_negative)):
        if (y_negative[i] < y_negative[i-1]) or (y_positive[i] < y_positive[i-1]):
            return False
    return True


def showStationary(fn, roots, maxima, minima):
    '''
    Given a function, a list of roots, list of maximas and a list of minimas, plots the function along with its stationary points
    returns None
    '''
    x = np.linspace(-3, 3, 10000)
    fx_roots = np.array(fn(sr_number, roots))
    fx_minima = fn(sr_number, np.array(minima))
    fx_maxima = fn(sr_number, np.array(maxima))
    fx = fn(sr_number, x)
    plt.plot(x, fx)
    plt.scatter(minima, fx_minima, color='red', label="Minima")
    plt.scatter(maxima, fx_maxima, color='violet', label="Maxima")
    plt.scatter(roots, fx_roots, color='green', label="Roots")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.show()


def getDerivative(fn, x):
    '''
    Given a funciton and a set of points in an interval, calculates the approximate derivative at each of those points
    return : List containting derivatives
    '''
    fx = [fn(sr_number, i) for i in x]
    h = 6/10000
    for i in range(len(fx)-1):
        fx[i] = (fx[i+1]-fx[i])/h
    return fx


def getSecondDx(fn, x):
    '''
    Given a funciton and a set of points in an interval, calculates the approximate second derivative at each of those points
    return : List containting second derivatives
    '''
    h = 6/10000
    d2 = (fn(sr_number, x+h) - 2*fn(sr_number, x) + fn(sr_number, x-h))/(h**2)
    return d2


def findRoots(fn):
    '''
    Given a function, calucates the roots by observing where the function value changes sign
    return list of roots
    '''

    x = np.linspace(-3, 3, 10000)
    fx = fn(sr_number, x)
    roots = []
    for i in range(len(fx)-1):
        if ((fx[i] < 0 and fx[i+1] > 0) or (fx[i] > 0 and fx[i+1] < 0)):
            roots.append((x[i]+x[i+1])/2)
    return np.array(roots)


def FindStationaryPoints(fn):
    '''
    Given a function, calculates stationary points by observing which derivatives are zero 
    distinguishes between them as maximas and minimas by calculating the second derivatives at those points
    returns dictionary contating roots, minimas, maximas
    '''
    x = np.linspace(-3, 3, 10000)
    gradx = getDerivative(fn, x)
    optima = []
    for i in range(len(x)):
        if (round(gradx[i], 2) == 0):
            optima.append(x[i])
    minima, maxima = [], []

    for item in optima:
        if (getSecondDx(fn, item) > 0):
            minima.append(item)
        elif (getSecondDx(fn, item) < 0):
            maxima.append(item)

    roots = findRoots(fn)
    return {"Roots": roots, "Minima": minima, "LocalMaxima": maxima}


'''
    Question 2
'''


def plotGradNorm(grad_norm):
    plt.plot([i for i in range(len(grad_norm))], grad_norm)
    plt.show()


def plotFValueDiff(f_value):
    plt.plot([i for i in range(len(f_value))], [
             i - f_value[-1] for i in f_value])
    plt.show()


def plotFValueDiffRatio(f_value):
    y = []
    for i in range(1, len(f_value)):
        y.append(0 if (f_value[i-1]-f_value[-1] == 0)
                 else (f_value[i]-f_value[-1])/(f_value[i-1]-f_value[-1]))

    plt.plot([i for i in range(1, len(f_value))], y)
    plt.show()


def plotPointDiffNorm(x):
    plt.plot([i for i in range(len(x))], [
             np.linalg.norm(item - x[-1])**2 for item in x])
    plt.show()


def plotPointDiffNormRatio(x):
    y = []
    for i in range(1, len(x)):
        y.append(0 if ((np.linalg.norm(x[i-1]-x[-1])**2) == 0) else (
            np.linalg.norm(x[i]-x[-1])**2)/(np.linalg.norm(x[i-1]-x[-1])**2))

    plt.plot([i for i in range(1, len(x))], y)
    plt.show()


def ConstantGradientDescent(alpha, initialx, iterations=10000):
    '''
    Given initial alpha and x, constatnt gradient descent is calculated
    returns final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        fx, grad = f4(sr_number, x)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(fx)
        trajectory.append(x)
        x = x - alpha * grad

    return x, f_value, trajectory, grad_norm


def DiminishingGradientDescent(InititalAlpha, initialx, iterations=10000):
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
        fx, grad = f4(sr_number, x)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(fx)
        trajectory.append(x)
        x = x - alpha_0/(i+1) * grad

    return x, f_value, trajectory, grad_norm


def InExactLineSearch(c1, c2, gamma, initialx, iterations=5000):
    '''
    Given c1, c2, gamma, and initial x Inexact line search is calculated
    return final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    alpha = 1
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        fx, grad = f4(sr_number, x)
        fx_new, fx_grad_new = f4(sr_number, x+alpha*(-grad))
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(fx)
        trajectory.append(x)
        if (fx_new > fx + c1*alpha*np.dot(-grad, grad)) or (np.dot(grad, fx_grad_new) > -c2*np.dot(-grad, grad)):
            alpha = gamma*alpha
        x = x - alpha * grad

    return x, f_value, trajectory, grad_norm


def getExactAlpha(x, grad):
    '''
    Utility function for ExactLineSearch
    Caluclates the exact alpha given a list of x and gradient at x
    returns alpha: float
    '''
    _, grad_p = f4(sr_number, -grad)
    _, grad_px = f4(sr_number, x-grad)
    numerator = -np.dot(np.transpose(grad), -grad)
    denominator = np.dot(np.transpose(-grad), grad_p+grad_px)
    return 0 if denominator == 0 else numerator/denominator


def ExactLineSearch(initialx, iterations=100):
    '''
    Caluclates exact line search using getExactAlpha
    returns final x, list of funciton values, list of xs, list of gradient norms at each step
    '''
    x = initialx
    grad_norm = []
    f_value = []
    trajectory = []
    for _ in range(iterations):
        fx, grad = f4(sr_number, x)
        grad_norm.append(np.linalg.norm(grad))
        f_value.append(fx)
        trajectory.append(x)
        alpha = getExactAlpha(x, grad)
        x = x - alpha * grad

    return x, f_value, trajectory, grad_norm


'''
Question 3
'''


def fn(X):
    return np.e**(X[0]*X[1])


def grad_fn(X):
    return np.array([X[1]*np.e**(X[0]*X[1]), X[0]*np.e**(X[0]*X[1])])


def contourPlot(fn):
    X, Y = np.meshgrid(np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000))
    fx = fn([X, Y])
    fig, ax = plt.subplots()
    contour = plt.contour(X, Y, fx, colors='black',
                          linestyles='dashed', linewidths=1)
    plt.clabel(contour, inline=1, fontsize=10)
    contour = ax.contourf(X, Y, fx, cmap='YlGnBu')
    ax.set_aspect("equal")
    cbar = fig.colorbar(contour)
    cbar.ax.set_ylabel("Function value at x")
    plt.xlabel("X-values")
    plt.ylabel("Y-values")
    plt.show()


def plotFunctionValue(f_value):
    plt.plot([i for i in range(len(f_value))], f_value)
    plt.xlabel("Iterations")
    plt.ylabel("Function value")
    plt.show()


def plotTrajectory(trajectory):
    trajectory = np.array(trajectory)
    x = 1.5  # to reproduce graphs for Q3.4 , Q3.5, Q3.8 change x to 1.5 (4)
    X, Y = np.meshgrid(np.linspace(-x, x, 1000), np.linspace(-x, x, 1000))
    fx = fn([X, Y])
    fig, ax = plt.subplots()
    contour = plt.contour(X, Y, fx, colors='black',
                          linestyles='dashed', linewidths=1)
    plt.clabel(contour, inline=1, fontsize=10)
    contour = ax.contourf(X, Y, fx, cmap='YlGnBu')
    ax.set_aspect("equal")
    plt.scatter(trajectory[:, 0], trajectory[:, 1],
                color='red', marker='.', label='Gradient Descent Path')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1],
                color='blue', label='Optimum')
    cbar = fig.colorbar(contour)
    cbar.ax.set_ylabel("Function value at x")
    plt.legend()
    plt.xlabel("X-values")
    plt.ylabel("Y-values")
    plt.show()


def ConstantGradientDescentMultivariable(alpha, initialx, iterations=100000):
    '''Same as Question 2, but the function does not take sr_number as an argument'''
    x = initialx
    f_value = []
    trajectory = []
    for _ in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        x = x - alpha*grad
    return x, f_value, trajectory


def DiminishingGradientDescentMultivariable(alpha, initialx, iterations=100000):
    '''Same as Question 2, but the function does not take sr_number as an argument'''
    x = initialx
    alpha_0 = alpha
    f_value = []
    trajectory = []
    for i in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        alpha = alpha_0/(i+1)
        x = x - alpha*grad
    return x, f_value, trajectory


def PerturbedConstantBothGD(alpha, initialx, iterations=100000):
    '''
    Gaussian noise is added to the gradient to escape saddle points
    '''
    x = initialx
    f_value = []
    trajectory = []
    for i in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        x = x - alpha * \
            (grad+np.random.multivariate_normal([0, 0], np.identity(2)))
    return x, f_value, trajectory


def PerturbedDecreasingNoiseGD(alpha, initialx, var, iterations=100000):
    x = initialx
    f_value = []
    trajectory = []
    for i in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        x = x - alpha * \
            (grad +
             np.random.multivariate_normal([0, 0], (var/(i+1))*np.identity(2)))
    return x, f_value, trajectory


def PerturbedDecreasingStepGD(alpha, initialx, var, iterations=100000):
    x = initialx
    f_value = []
    trajectory = []
    for i in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        x = x - (alpha/(i+1)) * \
            (grad+np.random.multivariate_normal([0, 0], var*np.identity(2)))
    return x, f_value, trajectory


def PerturbedDecreasingBothGD(alpha, initialx, var, iterations=100000):
    x = initialx
    f_value = []
    trajectory = []
    for i in range(iterations):
        fx, grad = fn(x), grad_fn(x)
        f_value.append(fx)
        trajectory.append(x)
        x = x - (alpha/(i+1))*(grad +
                               np.random.multivariate_normal([0, 0], (var/(i+1))*np.identity(2)))
    return x, f_value, trajectory


'''
Question 4

'''


def plotGSFValue(f, x):
    plt.plot([i for i in range(1, len(x)+1)], [f(i) for i in x])
    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.show()


def plotGSbma(intervals):
    plt.plot([i for i in range(1, len(intervals)+1)], [i[1] - i[0]
             for i in intervals])
    plt.xlabel("Iterations")
    plt.ylabel("b_t - a_t")
    plt.show()


def plotGSRatio(intervals):
    y = [0 if (intervals[i-1][1] - intervals[i-1][0]) == 0 else (intervals[i][1] - intervals[i]
                                                                 [0])/(intervals[i-1][1] - intervals[i-1][0]) for i in range(1, len(intervals))]
    plt.plot([i for i in range(1, len(intervals))], y)
    plt.xlabel("Iterations")
    plt.ylabel("(b_t - a_t)/ (b_t-1 - a_t-1)")
    plt.show()


def f(x):
    return x*(x-1)*(x-3)*(x+2)


def GoldenSectionSearch(fn, interval, precision=1e-4):
    '''
    Uses golden section search to find an interval less than the precision we want that contains the minimum function value
    returns final interval and a list containing all intervals
    '''
    phi = (1+5**0.5)/2
    rho = phi - 1
    x1 = rho * interval[0] + (1-rho) * interval[1]
    x2 = (1-rho) * interval[0] + rho * interval[1]
    intervals = [interval[0], x1, x2, interval[1]]
    intervalList = []

    while (abs(intervals[3] - intervals[1]) > precision):
        intervalList.append([intervals[0], intervals[3]])
        if (fn(intervals[1]) <= fn(intervals[2])):
            x3 = rho*intervals[0] + (1-rho) * intervals[2]
            intervals = [intervals[0], x3, intervals[1], intervals[2]]
        else:
            x3 = (1-rho)*intervals[1] + rho * intervals[3]
            intervals = [intervals[1], intervals[2], x3, intervals[3]]
    return intervals, intervalList


def generateFibs(interval, precision=1e-4):
    '''
    utility funciton for FibonacciSearch
    returns a list of fibonacci numbers
    '''
    fibs = [0, 1]
    while ((interval[1] - interval[0])/precision > fibs[-1]):
        fibs.append(fibs[-1]+fibs[-2])
    return fibs


def FibonacciSearch(fn, interval, precision=1e-4):
    '''
    Uses fibonacci section search to find an interval less than the precision we want that contains the minimum function value
    returns final interval and a list containing all intervals
    '''
    fibs = generateFibs(interval)
    t_1, t_2 = -2, -1
    rho = 1 - (fibs[t_1]/fibs[t_2])
    t_1 -= 1
    t_2 -= 1
    x1 = (1-rho) * interval[0] + rho * interval[1]
    x2 = rho * interval[0] + (1-rho) * interval[1]
    intervals = [interval[0], x1, x2, interval[1]]
    intervalList = []

    while (abs(intervals[3] - intervals[1]) > precision):
        intervalList.append([intervals[0], intervals[3]])
        rho = 1 - (fibs[t_1]/fibs[t_2])
        t_1 -= 1
        t_2 -= 1
        if (fn(intervals[1]) <= fn(intervals[2])):
            x3 = (1-rho)*intervals[0] + rho * intervals[2]
            intervals = [intervals[0], x3, intervals[1], intervals[2]]
        else:
            x3 = rho*intervals[1] + (1-rho) * intervals[3]
            intervals = [intervals[1], intervals[2], x3, intervals[3]]
    return intervals, intervalList


def main():
    '''
    Question 1

    '''
    interval = [-2., 2.]
    print(isConvex(f1, interval))
    print(isConvex(f2, interval))
    print(isStrictlyConvex(f1, interval))
    print(isStrictlyConvex(f2, interval))
    print(findMinimas(f1))
    print(findMinimas(f2))
    print(isCoercive(f3))
    dict_points = FindStationaryPoints(f3)
    showStationary(
        f3, dict_points["Roots"], dict_points["LocalMaxima"], dict_points["Minima"])

    '''
    Question 2
    '''
    x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    x, f_value, trajectory, grad_norm = ConstantGradientDescent(1e-5, x_0)
    print(f"x: {x}, fx: {f_value[-1]}")
    plotGradNorm(grad_norm)
    plotFValueDiff(f_value)
    plotFValueDiffRatio(f_value)
    plotPointDiffNorm(trajectory)
    plotPointDiffNormRatio(trajectory)

    x, f_value, trajectory, grad_norm = DiminishingGradientDescent(1e-5, x_0)
    print(f"x: {x}, fx: {f_value[-1]}")
    plotGradNorm(grad_norm)
    plotFValueDiff(f_value)
    plotFValueDiffRatio(f_value)
    plotPointDiffNorm(trajectory)
    plotPointDiffNormRatio(trajectory)

    x, f_value, trajectory, grad_norm = InExactLineSearch(
        1e-4, 1-1e-4, 1e-5, x_0)
    print(f"x: {x}, fx: {f_value[-1]}")
    plotGradNorm(grad_norm)
    plotFValueDiff(f_value)
    plotFValueDiffRatio(f_value)
    plotPointDiffNorm(trajectory)
    plotPointDiffNormRatio(trajectory)

    x, f_value, trajectory, grad_norm = ExactLineSearch(x_0)
    print(f"x: {x}, fx: {f_value[-1]}")
    plotGradNorm(grad_norm)
    plotFValueDiff(f_value)
    plotFValueDiffRatio(f_value)
    plotPointDiffNorm(trajectory)
    plotPointDiffNormRatio(trajectory)

    '''
    Question 3
    '''
    contourPlot(fn)

    x, f_value, trajectory = ConstantGradientDescentMultivariable(
        1e-4, [1.2, 1.2])
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    x, f_value, trajectory = DiminishingGradientDescentMultivariable(
        1e-3, [1.2, 1.2])
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    x, f_value, trajectory = PerturbedConstantBothGD(1e-3, [1.2, 1.2])
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    x, f_value, trajectory = PerturbedDecreasingNoiseGD(1e-3, [1.2, 1.2], 1)
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    x, f_value, trajectory = PerturbedDecreasingStepGD(1e-3, [1.2, 1.2], 1)
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    x, f_value, trajectory = PerturbedDecreasingBothGD(1e-3, [1.2, 1.2], 1)
    print(f"x: {x}, fx: {fn(x)}")
    plotFunctionValue(f_value)
    plotTrajectory(trajectory)

    '''
    Question 4
    '''

    _, intervals = GoldenSectionSearch(f, [1, 3])
    plotGSFValue(f, [i[0] for i in intervals])
    plotGSFValue(f, [i[1] for i in intervals])
    plotGSbma(intervals)
    plotGSRatio(intervals)

    _, intervals = FibonacciSearch(f, [1, 3])
    plotGSFValue(f, [i[0] for i in intervals])
    plotGSFValue(f, [i[1] for i in intervals])
    plotGSbma(intervals)
    plotGSRatio(intervals)


if __name__ == "__main__":
    main()
