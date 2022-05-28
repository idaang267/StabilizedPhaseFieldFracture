import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt

# Here x = independent variable and theta = [K, r, t0]
def LogFcn(x, theta):
    return theta[0] / (1 + np.exp(- theta[1] * (x - theta[2])))

# Define residual for nonlinear fitting
def ResLog(theta, x, y):
    return y - LogFcn(x, theta)

def LogLoading(savedir, LoadMax, LoadSteps, Inc, IntPoint, LogGrowthRate):
    int_point = LoadSteps/IntPoint        # Intermediate Point
    # Values for logistic
    K = LoadMax         # Supremum of function y, maximum achieved when x = inf
    r = LogGrowthRate   # logistic growth rate or sharpness of curve
    t0 = int_point      # value at which the transition point is achieved
    # Start for optimizer
    theta0 = [LoadMax, 0.1, t0]
    # Set up evenly spaced time steps
    EvenSteps = np.linspace(0, LoadSteps, LoadSteps)
    # Set up a few points to define the desired function
    Nsteps = np.array([0, 1, 2, int_point, LoadSteps-3, LoadSteps-2, LoadSteps-1, LoadSteps])
    Delta = np.array([0, Inc, 2*Inc, LoadMax/2, LoadMax-3*Inc, LoadMax-2*Inc, LoadMax-Inc, LoadMax])
    # Least square regression
    popt1, pcov1 = opt.leastsq(ResLog, theta0, args=(Nsteps, Delta))
    LoadMultipliers = LogFcn(EvenSteps, [*popt1])

    plt.figure()
    h1 = plt.plot(Nsteps, Delta, 'k.', label='Raw Data')
    h4 = plt.plot(EvenSteps, LoadMultipliers, 'b:', label='LSQ Approach: Log')
    plt.legend(loc="best", frameon=False)
    plt.xlabel("Number of Steps")
    plt.ylabel("Function")
    plt.savefig(savedir + 'LoadingFit.pdf', transparent=True)
    plt.close()
    return LoadMultipliers
