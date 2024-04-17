from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import bernoulli_nb_numpyro as bnb


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data():
    # Generate a synthetic binary classification dataset.
    # n_samples: The total number of samples to generate.
    # n_features: The total number of features. These include informative, redundant, and random noise features.
    # n_informative: The number of informative features, i.e., features actually used to build the class labels.
    # n_redundant: The number of redundant features, i.e., random linear combinations of the informative features.
    # n_classes: The number of classes (or labels) for the classification.
    # random_state: Controls the shuffling applied to the data before applying the split for reproducibility.
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, n_classes=2, random_state=42)
    
    # Convert the features into binary (0 or 1) using thresholding.
    # Here, positive numbers are converted to 1 and non-positive numbers are converted to 0.
    X = (X > 0).astype(int)

    return X, y

# Generate dataset
X, y = generate_data()

# Split the dataset into training and testing sets.
# test_size: The proportion of the dataset to include in the test split (30% in this case).
# random_state: Controls the shuffling applied to the data before the split for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.naive_bayes import BernoulliNB

def run_bernoulli_nb_sklearn(X_train, X_test, y_train, y_test):
    # Create an instance of the BernoulliNB classifier.
    # This classifier is particularly suited for discrete data, especially binary/boolean features.
    clf = BernoulliNB()
    
    # Fit the Bernoulli Naive Bayes classifier on the training data.
    # X_train: Training feature matrix.
    # y_train: Training response vector (class labels).
    clf.fit(X_train, y_train)
    
    # Predict the probability of the positive class for the first 5 samples in the test set.
    # This method returns the probability estimates for all classes for each instance.
    # [:,1] is used to slice the probabilities of the positive class (class 1) from the output.
    print(
        clf.predict_proba(X_test[:5])[:,1]
    )

# Running the function with training and testing data.
# This function call utilizes the training and testing datasets to train the model and then print 
# predicted probabilities for a subset of the testing dataset.
run_bernoulli_nb_sklearn(X_train, X_test, y_train, y_test)

# Output:
# [0.54081163 0.18180673 0.1106803  0.77287985 0.96626487]

# Train the Bernoulli Naive Bayes model using NumPyro with the defined Bayesian model.
# The training data (X_train, y_train) is passed to the model.
# The function returns the posterior samples from the MCMC simulation, which are used for prediction.
samples = bnb.train_model(bnb.bernoulli_nb_model, X_train, y_train)

# Predict the probabilities of the positive class for the first 5 samples in the testing set using the trained model.
# The function 'predict_proba' uses the mean of the posterior samples to calculate these probabilities.
# The slicing [:,1] selects the probabilities corresponding to class 1 (positive class).
print(
    bnb.predict_proba(samples, X_test[:5])[:,1]
)

# Predict probability intervals for the same subset of the testing set.
# 'predict_proba_with_interval' calculates the lower and upper bounds of the prediction intervals for class 1 probabilities.
# The intervals are computed using the percentiles of the posterior sample distributions, which provides a measure of uncertainty.
print(
    bnb.predict_proba_with_interval(samples, X_test[:5])[:,1,:]
)

# Output shows the predicted probabilities and their intervals for class 1:
# The first array displays probabilities for class 1 of the first five test samples.
# The second array shows the confidence intervals (lower and upper bounds) for these probabilities.
# Example:
# Probabilities: [0.547578   0.18020964 0.10945402 0.77462715 0.96614677]
# Intervals:
# [[0.24342528 0.82694507]
#  [0.0447402  0.43090346]
#  [0.02792111 0.29280424]
#  [0.5043225  0.93098795]
#  [0.90318424 0.9932422 ]]
