import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc import HalfCauchy, Model, Normal, sample

print(f"Running on PyMC v{pm.__version__}")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


az.style.use("arviz-darkgrid")

def main():
    size = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + rng.normal(scale=0.5, size=size)

    data = pd.DataFrame(dict(x=x, y=y))
    
    with Model() as model1:  # model specifications in PyMC are wrapped in a with-statement
        # Define priors
        sigma = HalfCauchy("sigma", beta=10)
        intercept = Normal("intercept", 0, sigma=20)
        slope = Normal("slope", 0, sigma=20)

        # Define likelihood
        likelihood = Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)
    
    """
    with Model() as model2:
        # Define priors
        sigma = Normal("sigma", 0,sigma=1)
        intercept = Normal("intercept", 0, sigma=20)
        slope = Normal("slope", 0, sigma=20)

        # Define likelihood
        likelihood = Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)
    """
    
    #print(model1["sigma","slope"])
    with model1:
        #stepmethod=[pm.NUTS([model1["sigma"],model1["slope"],model1["intercept"]])]
        step1=pm.NUTS([model1["sigma"]])
        step2=pm.NUTS([model1["slope"],model1["intercept"]])
        stepmethod=[step1,step2]
        samples1=sample(1000,step=stepmethod)
        #samples1=sample(1000)
    
    """
    with model2:
        stepmethod=[pm.NUTS([model2["sigma"]]),pm.NUTS([model2["sigma"],model2["slope"]]), pm.NUTS([model2["sigma"],model2["slope"],model2["intercept"]])]

        samples2=sample(1000,step=stepmethod)
    """

    #print(samples1['posterior'])

    az.plot_trace(data=samples1,var_names=["sigma","slope","intercept"],divergences=True)
    az.plot_energy(data=samples1)
    az.plot_pair(samples1, var_names=["sigma","slope","intercept"], divergences=True)
    print(az.summary(data=samples1,var_names=["sigma","slope","intercept"]))
    
if __name__=='__main__':
    main()