import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

def bernoulli_nb_model(X, y=None):
    # Get the number of samples and features from the input dataset.
    n_samples, n_features = X.shape

    # Define the class labels. Assuming binary classification (0 and 1).
    classes = jnp.array([0, 1])

    # Count the number of classes.
    n_classes = len(classes)
    
    # Define the prior distribution for the class probabilities.
    # A Dirichlet distribution is used because it is a multivariate generalization of the beta distribution,
    # appropriate for modeling probabilities of a multinomial distribution (multiple classes).
    class_probs = numpyro.sample("class_probs", dist.Dirichlet(jnp.ones(n_classes)))
    
    # Define the prior distributions for the probabilities of features given the class.
    # A Beta distribution is used for modeling the probability of binary outcomes.
    # Separate probabilities are modeled for each feature and each class.
    feature_probs = numpyro.sample("feature_probs", dist.Beta(jnp.ones((n_classes, n_features)),
                                                               jnp.ones((n_classes, n_features))))
    
    # Sample the expected class labels (y_hat) using the categorical distribution based on class probabilities.
    # This represents our belief about the most likely class before observing the features.
    y_hat = numpyro.sample("y_hat", dist.Categorical(probs=class_probs), obs=y)
    
    # Model each feature as a Bernoulli distributed variable, conditioned on the expected class.
    # The loop iterates over each feature to model its probability conditional on the class derived from y_hat.
    for i in range(n_features):
        numpyro.sample(f"X_{i}", dist.Bernoulli(probs=feature_probs[y_hat, i]), obs=X[:, i])

def train_model(model, X, y):
    # Set up the No-U-Turn Sampler (NUTS), which is an efficient and adaptive Hamiltonian Monte Carlo (HMC) method.
    kernel = NUTS(model)
    
    # Configure and execute the Markov Chain Monte Carlo (MCMC) simulation.
    # num_warmup: Number of warm-up steps used to tune the MCMC algorithm.
    # num_samples: Number of samples to draw from the posterior distribution after warm-up.
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    
    # Run the MCMC algorithm with the given model, inputs (X), and labels (y).
    # jax.random.PRNGKey(0) provides a random seed to ensure reproducibility.
    mcmc.run(jax.random.PRNGKey(0), X=X, y=y)
    
    # Extract and return the sampled posterior distributions.
    return mcmc.get_samples()


def calculate_class_prob(class_probs, feature_probs, X_new):
    # Calculate log probabilities of features given class using matrix operations.
    # jnp.log(feature_probs.T) calculates the log of the probabilities of each feature being 1 given the class.
    # (1 - X_new) @ jnp.log(1 - feature_probs.T) computes the log probabilities of features being 0.
    # The result is summed to find the total log probability of the features given the class.
    log_prob_X_given_class = X_new @ jnp.log(feature_probs.T) + (1 - X_new) @ jnp.log(1 - feature_probs.T)
    
    # Add the log probabilities of the classes to the log probabilities of the features given the class.
    # This step integrates class priors with the evidence from the features.
    log_prob = log_prob_X_given_class + jnp.log(class_probs)
    
    # Convert log probabilities back to normal probabilities by exponentiating and normalizing.
    # Subtracting the max log probability before exponentiation helps avoid numerical underflow.
    prob = jnp.exp(log_prob - jnp.max(log_prob, axis=1, keepdims=True))
    
    # Normalize the probabilities so that each row sums to 1, representing valid probability distributions.
    class_probabilities = prob / jnp.sum(prob, axis=1, keepdims=True)
    
    return class_probabilities

def predict_proba(samples, X_new):
    # Compute the mean class and feature probabilities from the posterior samples.
    # This averages over all sampled estimates to get a single probability estimate for each class and feature.
    class_probs = jnp.mean(samples['class_probs'], axis=0)
    feature_probs = jnp.mean(samples['feature_probs'], axis=0)
    
    # Use the mean probabilities to calculate class probabilities for new data points.
    return calculate_class_prob(class_probs, feature_probs, X_new)

def predict_proba_with_interval(samples, X_new, alpha=0.05):
    # Extract arrays of class and feature probabilities from the samples.
    class_probs_samples = samples['class_probs']
    feature_probs_samples = samples['feature_probs']
    
    # Determine the number of samples and classes from the shape of the sampled data.
    n_samples = len(class_probs_samples)
    n_classes = class_probs_samples.shape[1]
    
    # Initialize a list to store class probabilities for each sample.
    all_class_probs = []
    for i in range(n_samples):
        # For each sample, calculate the class probabilities for new data points.
        class_probs = class_probs_samples[i]
        feature_probs = feature_probs_samples[i]
        p = calculate_class_prob(class_probs, feature_probs, X_new)
        all_class_probs.append(p)
    
    # Convert the list of class probabilities into a NumPy array for vectorized operations.
    all_class_probs = jnp.array(all_class_probs)
    
    # Calculate the lower and upper percentiles to form the confidence intervals for each class prediction.
    lower = jnp.percentile(all_class_probs, alpha / 2 * 100, axis=0)
    upper = jnp.percentile(all_class_probs, (1 - alpha / 2) * 100, axis=0)
    
    # Build intervals for each data point across each class.
    intervals = []
    for i in range(X_new.shape[0]):
        intervals.append([
            [lower[i][0], upper[i][0]],  # Interval for class 0
            [lower[i][1], upper[i][1]]  # Interval for class 1
        ])
    intervals = jnp.array(intervals)
    
    return intervals


    
