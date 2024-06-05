
### 1. Bayesian Optimization
from bayes_opt import BayesianOptimization

def bayesian_optimize(my_function, bounds**, init_points=5, n_iter=25):
    ''' 
    Parameters
    ----------
    function : input function to optimise, must return a single value to maximise
    bounds** : dictionary, containing bounds for each list of params
    init_points : starting point for optimiser
    n_iter : iterations of optimiser
    
    Returns
    -------
    Optimised Values.
    
    Example
    -------
    def my_function(discrete_param, list_param, continuous_param):
        # ... your function logic
        return some_value

    # Define the bounds for your parameters
    # For discrete parameters, you can specify the values directly
    # For continuous parameters, specify a range
    # For list parameters, you might need to encode them or treat them as discrete
    bounds = {'discrete_param': (0, 10),
              'list_param': ['A', 'B', 'C'],
              'continuous_param': (0, 1)}

    '''
    # Initialize the Bayesian Optimization
    optimizer = BayesianOptimization(f=my_function,
                                     pbounds=bounds,
                                     verbose=2,
                                     random_state=42)

    # Set the number of iterations
    optimizer.maximize(init_points=5, n_iter=25)

    return optimizer.max






### 2. Genetic Algorithms
import numpy as np
from deap import base, creator, tools, algorithms

def genetic_algorithm_optimiser(my_function):
    # Define your evaluation function
    def evaluate(individual):
        discrete_param, list_param, continuous_param = individual
        # Call your function with the individual's parameters
        return my_function(discrete_param, list_param, continuous_param)

    # Initialize DEAP
    toolbox = base.Toolbox()
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define how to create an individual
    toolbox.register("attr_discrete", random.randint, 0, 10)
    toolbox.register("attr_list", random.choice, ['A', 'B', 'C'])
    toolbox.register("attr_continuous", random.uniform, 0, 1)
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_discrete, toolbox.attr_list, toolbox.attr_continuous), n=1)

    # Define the population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation operator
    toolbox.register("evaluate", evaluate)

    # Register the crossover and mutation operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create an initial population
    pop = toolbox.population(n=300)

    # Run the Genetic Algorithm
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof)

    # The best individual is in the Hall of Fame
    best_individual = hof[0]
    return best_individual.fitness.values[0], best_individual

###############################################################
#Carry on from here


################################

### 3. Hyperopt
import hyperopt
from hyperopt import hp, fmin, tpe, Trials

# Define your objective function
def objective(params):
    discrete_param = int(params['discrete_param'])
    list_param = params['list_param']
    continuous_param = params['continuous_param']
    # Call your function with the parameters
    score = my_function(discrete_param, list_param, continuous_param)
    # Hyperopt minimizes, so return negative of score if you want to maximize
    return -score

# Define the search space
space = {
    'discrete_param': hp.quniform('discrete_param', 0, 10, 1), # Discrete parameters
    'list_param': hp.choice('list_param', ['A', 'B', 'C']), # Categorical parameters
    'continuous_param': hp.uniform('continuous_param', 0, 1) # Continuous parameters
}

# Initialize the trials object
trials = Trials()

# Run the optimization
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=100)

# The best parameters found
print(f"Best parameters found: {best}")


### 4. Differential_evolution

```python
import numpy as np
from scipy.optimize import differential_evolution

# Define your function (for example purposes, I'll create a dummy function)
def my_function(params):
    discrete_param, categorical_index, continuous_param = params
    categorical_param = ['A', 'B', 'C'][int(categorical_index)] # Map index to categorical value
    # This is a dummy calculation, replace with your actual function logic
    return - (discrete_param**2 + continuous_param**2) if categorical_param == 'B' else - (discrete_param**2 + continuous_param**2) * 1.5

# Define the bounds for each parameter
# Note: For discrete parameters, you can specify the bounds tightly around integers
# For categorical parameters, you can use an integer to represent the index of the category and then map it inside your function
bounds = [(0, 10), # Discrete parameter bounds
          (0, 2), # Categorical parameter bounds (index 0, 1, or 2)
          (0, 1)] # Continuous parameter bounds

# Define a function to handle the constraints (optional)
# In this case, we don't have constraints, but this is how you would define them
def constraint(x):
    return np.array([x[0] - np.round(x[0])]) # Ensure the discrete parameter is an integer

# Use differential_evolution for optimization
# We'll minimize the negative of the function since differential_evolution minimizes
result = differential_evolution(lambda x: -my_function(x), bounds, constraints=[{'type': 'eq', 'fun': constraint}])

# The best parameters found
best_params = result.x
best_discrete_param = int(best_params[0])
best_categorical_param = ['A', 'B', 'C'][int(best_params[1])]
best_continuous_param = best_params[2]

print(f"Best parameters found: discrete_param={best_discrete_param}, categorical_param={best_categorical_param}, continuous_param={best_continuous_param}")
print(f"Maximum value of the function: {-result.fun}")
```

In this example, `differential_evolution` is used to maximize `my_function` by minimizing its negative. The function has a discrete parameter, a categorical parameter represented by an integer index, and a continuous parameter. The bounds are defined accordingly, and a constraint is added to ensure the discrete parameter remains an integer.

Remember, `differential_evolution` is a global optimizer and can be quite effective for a wide range of problems, including those with mixed parameter types. However, it's essential to scale and bound your parameters correctly to achieve the best performance.

For discrete parameters, you might sometimes see an approach where the bounds are set to include only the desired integer values, but this can lead to issues if the optimizer tries to sample between those bounds. An alternative is to round the discrete parameters within the function itself, but this can affect the optimizer's ability to converge. The constraint approach shown here is a compromise that guides the optimizer towards integer values for discrete parameters.

For categorical parameters, the common approach is to use an integer to represent the category's index, as shown in the example. This method is simple and effective, especially when the number of categories is small.

Always consider the nature of your problem and the behavior of your function when choosing an optimization strategy. `differential_evolution` is a powerful tool, but it's one of many available, and the best choice can vary depending on the specifics of your optimization task.