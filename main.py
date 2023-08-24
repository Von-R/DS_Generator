import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from sklearn.utils import resample
from typing import Optional, List


def generate_numerical_attributes(num_rows: int, num_columns: int, mean: Optional[List[float]] = None,
                                  std_dev: Optional[List[float]] = None, skewness: Optional[List[float]] = None,
                                  min_range: Optional[List[float]] = None,
                                  max_range: Optional[List[float]] = None) -> np.ndarray:
    """
    Generates numerical attributes with given parameters.

    :param num_rows: Number of rows in the dataset
    :param num_columns: Number of numerical columns
    :param mean: List of mean values for each column
    :param std_dev: List of standard deviations for each column
    :param skewness: List of skewness values for each column
    :param min_range: List of minimum values for each column
    :param max_range: List of maximum values for each column
    :return: A numpy array containing the generated numerical attributes
    """
    numerical_data = np.zeros((num_rows, num_columns))

    for col, (col_mean, col_std_dev, col_skewness, col_min, col_max) in enumerate(zip(
            mean or [np.random.uniform(10, 50) for _ in range(num_columns)],
            std_dev or [np.random.uniform(5, 25) for _ in range(num_columns)],
            skewness or [np.random.uniform(-2, 2) for _ in range(num_columns)],
            min_range or [np.random.uniform(0, 30) for _ in range(num_columns)],
            max_range or [np.random.uniform(35, 100) for _ in range(num_columns)])):
        col_data = skewnorm.rvs(col_skewness, loc=col_mean, scale=col_std_dev, size=num_rows)
        col_data = np.clip(col_data, col_min, col_max)
        numerical_data[:, col] = col_data

    return numerical_data


def generate_categorical_attributes(num_rows, num_columns, num_categories=None, category_labels=None):
    if num_categories is None:
        num_categories = np.random.randint(2, 6, num_columns)

    categorical_attributes = np.empty((num_rows, num_columns), dtype=object)
    for col in range(num_columns):
        if category_labels and len(category_labels) > col:
            labels = category_labels[col]
        else:
            labels = [str(i) for i in range(num_categories[col])]

        categorical_attributes[:, col] = np.random.choice(labels, size=num_rows)

    return categorical_attributes


def combine_attributes(num_rows, numerical_data, categorical_data, column_names=None):
    numerical_columns = [f'num_{i}' for i in range(numerical_data.shape[1])]
    numerical_df = pd.DataFrame(numerical_data, columns=numerical_columns)
    categorical_columns = [f'cat_{i}' for i in range(categorical_data.shape[1])]
    categorical_df = pd.DataFrame(categorical_data, columns=categorical_columns)
    dataset = pd.concat([numerical_df, categorical_df], axis=1)

    return dataset


def add_impurities(dataset, missing_percentage=5, num_outliers=2, noise_std_dev=0.5):
    training_dataset = dataset.copy()
    if missing_percentage > 0:
        num_missing = int(missing_percentage * dataset.size / 100)
        for _ in range(num_missing):
            row_idx = np.random.randint(dataset.shape[0])
            col_idx = np.random.randint(dataset.shape[1])
            training_dataset.iat[row_idx, col_idx] = None

    numerical_columns = dataset.select_dtypes(include=['float64']).columns
    for _ in range(num_outliers):
        row_idx = np.random.randint(dataset.shape[0])
        col_idx = np.random.choice(numerical_columns)
        outlier_value = np.random.uniform(dataset[col_idx].min() * 2, dataset[col_idx].max() * 2)
        training_dataset.at[row_idx, col_idx] = outlier_value

    for col in numerical_columns:
        noise = np.random.normal(0, noise_std_dev, training_dataset.shape[0])
        training_dataset[col] += noise

    return training_dataset


def extrapolate_dataset(dataset, size):
    return resample(dataset, replace=True, n_samples=size)


def generate_training_from_uploaded(test_dataset, extrapolate_size=None, missing_percentage=5, num_outliers=2,
                                    noise_std_dev=0.5):
    if extrapolate_size and extrapolate_size > len(test_dataset):
        test_dataset = extrapolate_dataset(test_dataset, extrapolate_size)

    training_dataset = add_impurities(test_dataset, missing_percentage, num_outliers, noise_std_dev)
    return training_dataset


def generate_correlated_numerical_attributes(num_rows, means, std_devs, skewness, min_range, max_range,
                                             correlation_matrix):
    # Generate multivariate normal data with the given correlation structure
    print("std_devs shape:", np.shape(std_devs))
    print("correlation_matrix shape:", np.shape(correlation_matrix))
    covariance_matrix = np.diag(std_devs) @ correlation_matrix @ np.diag(std_devs)
    data = np.random.multivariate_normal(means, covariance_matrix, size=num_rows)

    # Apply skewness transformation
    for col in range(data.shape[1]):
        data[:, col] = skewnorm.rvs(skewness[col], loc=means[col], scale=std_devs[col], size=num_rows)

    # Scale the data to fit within the specified range
    for col in range(data.shape[1]):
        min_val = np.min(data[:, col])
        max_val = np.max(data[:, col])
        data[:, col] = min_range[col] + (data[:, col] - min_val) * (max_range[col] - min_range[col]) / (
                max_val - min_val)

    return data


def generate_correlated_categorical_attributes(num_rows, categories_A, categories_B, conditional_prob_B_given_A):
    categorical_data_A = np.random.choice(categories_A, size=num_rows)
    categorical_data_B = []

    for value_A in categorical_data_A:
        prob_B = conditional_prob_B_given_A[value_A]
        value_B = np.random.choice(categories_B, p=prob_B)
        categorical_data_B.append(value_B)

    return np.column_stack((categorical_data_A, categorical_data_B))


def generate_random_correlation_matrix(num_columns):
    # Generate a random symmetric matrix
    random_matrix = np.random.uniform(-1, 1, (num_columns, num_columns))
    symmetric_matrix = (random_matrix + random_matrix.T) / 2

    # Ensure the diagonal elements are 1
    np.fill_diagonal(symmetric_matrix, 1)

    # Perform eigenvalue decomposition and adjust eigenvalues to ensure positive semi-definiteness
    eigenvalues, eigenvectors = np.linalg.eig(symmetric_matrix)
    adjusted_eigenvalues = np.maximum(eigenvalues, 0)
    correlation_matrix = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T

    return correlation_matrix


def generate_training_from_scratch(autogen=False):
    # Prompt for the number of rows and columns

    if autogen:
        num_rows = 100
        num_columns_numerical = 5
        correlated_numerical = '1'
        num_columns_categorical = 3
        correlated_categorical = '1'
        mean = np.random.uniform(10, 65, num_columns_numerical)
        std_dev = np.random.uniform(5, 20, num_columns_numerical)  # Example of random default values
        skew_stdev = np.random.uniform(0.2, 2, num_columns_numerical)
        skew_mean = np.random.uniform(-2, 2, num_columns_numerical)
        skewness = np.random.normal(skew_mean, skew_stdev, num_columns_numerical)
        min_range = np.random.uniform(0, 40, num_columns_numerical)
        max_range = np.random.uniform(45, 100, num_columns_numerical)
        correlation_matrix = generate_random_correlation_matrix(num_columns_numerical)
        numerical_data = generate_correlated_numerical_attributes(num_rows, mean, std_dev, skewness, min_range,
                                                                  max_range,
                                                                  correlation_matrix)
        categorical_scheme = np.random.randint(0, 6)

        if categorical_scheme <= 1:
            # One categorical variable with random number of categories
            num_categories_value = np.random.randint(2, 5)
            num_categories = [num_categories_value] * num_columns_categorical
            categories = ['cat_' + str(j) for j in range(num_categories_value)]
            print("One categorical variable with categories:", categories)
            categorical_data = generate_categorical_attributes(num_rows, num_columns_categorical, num_categories)
            # You can add further logic here for handling one categorical variable
        else:
            # Between 2 and N categorical variables; categories divided roughly equally among variables
            # Randomly determine the number of variables (between 1 and total_num_categories)
            num_variables = np.random.randint(1, num_columns_categorical + 1)

            # Randomly distribute categories among variables
            num_categories_per_variable = np.random.multinomial(num_columns_categorical,
                                                                [1 / num_variables] * num_variables)

            # Create a list of category labels for each variable
            categories_per_variable = [['var_' + str(i) + '_cat_' + str(j) for j in range(num_categories)] for
                                       i, num_categories in enumerate(num_categories_per_variable)]

            # Optionally, you could introduce dependencies between variables here In this example, we'll randomly
            # choose a dependent variable and its parent, and generate conditional probabilities
            if num_variables == 1:
                # Handle the case with only one variable by generating random samples from the categories
                categories = categories_per_variable[0]
                categorical_data = np.random.choice(categories, size=num_rows)
                # You can now use single_categorical_data as needed...
            else:
                # Handle the case with more than one variable (existing code)
                dependent_var_idx = np.random.randint(num_variables)
                parent_var_idx = np.random.choice([i for i in range(num_variables) if i != dependent_var_idx])

                categories_A = categories_per_variable[parent_var_idx]
                categories_B = categories_per_variable[dependent_var_idx]
                num_categories_B = num_categories_per_variable[dependent_var_idx]

                # Generate random conditional probabilities for B given A
                conditional_prob_B_given_A = {}
                for category_A in categories_A:
                    probabilities = np.random.rand(num_categories_B)
                    probabilities /= probabilities.sum()  # Normalize to sum to 1
                    conditional_prob_B_given_A[category_A] = probabilities.tolist()

                print("Two correlated categorical variables with conditional probabilities:",
                      conditional_prob_B_given_A)

                # Assuming generate_correlated_categorical_attributes function is defined elsewhere
                categorical_data = generate_correlated_categorical_attributes(1000, categories_A, categories_B,
                                                                              conditional_prob_B_given_A)
        combined_dataset = combine_attributes(num_rows, numerical_data, categorical_data)
        missing_percentage = np.random.randint(2, 11)
        num_outliers = int(round(num_rows * np.random.uniform(0, 0.04)))
        noise_std_dev = np.random.normal(0.5, np.random.uniform(0.1, 1))
        training_dataset = add_impurities(combined_dataset, missing_percentage, num_outliers, noise_std_dev)
        test_dataset = combined_dataset.copy()
        return training_dataset, test_dataset

    num_rows_input = input("Enter the number of rows: ")
    num_rows = int(num_rows_input) if num_rows_input else 100
    num_columns_numerical_input = input("Enter the number of numerical columns: ")
    num_columns_numerical = int(num_columns_numerical_input) if num_columns_numerical_input else 5
    if num_columns_numerical != 0:
        correlated_numerical = input("Generate correlated numerical attributes? 1) Yes 2) No: ")
    num_columns_categorical_input = input("Enter the number of categorical columns: ")
    num_columns_categorical = int(num_columns_categorical_input) if num_columns_categorical_input else 3
    if num_columns_categorical != 0:
        correlated_categorical = input("Generate correlated categorical attributes? 1) Yes 2) No: ")
    # Prompt for numerical attributes parameters
    # Prompt for mean values
    user_input = input(
        f"Enter mean values (comma-separated) for {num_columns_numerical} columns, or leave blank for random: ")
    if user_input:
        mean = [float(x) for x in user_input.split(",")]
    else:
        mean = np.random.uniform(10, 65, num_columns_numerical)

    # Prompt for standard deviation values
    user_input = input("Enter standard deviation values (comma-separated) or leave blank for random: ")
    if user_input:
        std_dev = [float(x) for x in user_input.split(",")]
    else:
        # Provide default values, such as random values or a specific default
        std_dev = np.random.uniform(5, 20, num_columns_numerical)  # Example of random default values

    # Prompt for skewness values
    user_input = input("Enter skewness values (comma-separated) or leave blank for random: ")
    if user_input:
        skewness = [float(x) for x in user_input.split(",")]
    else:
        skew_stdev = np.random.uniform(0.2, 2, num_columns_numerical)
        skew_mean = np.random.uniform(-2, 2, num_columns_numerical)
        skewness = np.random.normal(skew_mean, skew_stdev, num_columns_numerical)

    # Prompt for minimum range values
    user_input = input("Enter minimum values (comma-separated) or leave blank for random: ")
    if user_input:
        min_range = [float(x) for x in user_input.split(",")]
    else:
        min_range = np.random.uniform(0, 40, num_columns_numerical)

    # Prompt for maximum range values
    user_input = input("Enter maximum values (comma-separated) or leave blank for random: ")
    if user_input:
        max_range = [float(x) for x in user_input.split(",")]
    else:
        max_range = np.random.uniform(45, 100, num_columns_numerical)

    # Generate numerical attributes
    if correlated_numerical == '1':
        correlation_matrix = generate_random_correlation_matrix(num_columns_numerical)
        numerical_data = generate_correlated_numerical_attributes(num_rows, mean, std_dev, skewness, min_range,
                                                                  max_range,
                                                                  correlation_matrix)
    else:
        numerical_data = generate_numerical_attributes(num_rows, num_columns_numerical, mean, std_dev, skewness)

    # Prompt for categorical attributes parameters
    num_categories = [int(x) for x in input(
        "Enter the number of categories for each categorical column (comma-separated) or leave blank for random: ").split(
        ",")] if input(
        "Enter the number of categories for each categorical column (comma-separated) or leave blank for random: ") else None

    # Generate categorical attributes
    if correlated_categorical == '1':
        # Assuming correlated_categorical is input by the user or determined elsewhere
        categorical_scheme = np.random.randint(0, 6)

        if categorical_scheme <= 1:
            # One categorical variable with random number of categories
            num_categories_value = np.random.randint(2, 5)
            num_categories = [num_categories_value] * num_columns_categorical
            categories = ['cat_' + str(j) for j in range(num_categories_value)]
            print("One categorical variable with categories:", categories)
            categorical_data = generate_categorical_attributes(num_rows, num_columns_categorical, num_categories)
            # You can add further logic here for handling one categorical variable
        else:
            # Between 2 and N categorical variables; categories divided roughly equally among variables
            # Randomly determine the number of variables (between 1 and total_num_categories)
            num_variables = np.random.randint(1, num_columns_categorical + 1)

            # Randomly distribute categories among variables
            num_categories_per_variable = np.random.multinomial(num_columns_categorical,
                                                                [1 / num_variables] * num_variables)

            # Create a list of category labels for each variable
            categories_per_variable = [['var_' + str(i) + '_cat_' + str(j) for j in range(num_categories)] for
                                       i, num_categories in enumerate(num_categories_per_variable)]

            # Optionally, you could introduce dependencies between variables here In this example, we'll randomly
            # choose a dependent variable and its parent, and generate conditional probabilities
            if num_variables == 1:
                # Handle the case with only one variable by generating random samples from the categories
                categories = categories_per_variable[0]
                single_categorical_data = np.random.choice(categories, size=num_rows)
                # You can now use single_categorical_data as needed...
            else:
                # Handle the case with more than one variable (existing code)
                dependent_var_idx = np.random.randint(num_variables)
                parent_var_idx = np.random.choice([i for i in range(num_variables) if i != dependent_var_idx])

                categories_A = categories_per_variable[parent_var_idx]
                categories_B = categories_per_variable[dependent_var_idx]
                num_categories_B = num_categories_per_variable[dependent_var_idx]

                # Generate random conditional probabilities for B given A
                conditional_prob_B_given_A = {}
                for category_A in categories_A:
                    probabilities = np.random.rand(num_categories_B)
                    probabilities /= probabilities.sum()  # Normalize to sum to 1
                    conditional_prob_B_given_A[category_A] = probabilities.tolist()

                print("Two correlated categorical variables with conditional probabilities:",
                      conditional_prob_B_given_A)

                # Assuming generate_correlated_categorical_attributes function is defined elsewhere
                categorical_data = generate_correlated_categorical_attributes(1000, categories_A, categories_B,
                                                                              conditional_prob_B_given_A)


    else:
        categorical_data = generate_categorical_attributes(num_rows, num_columns_categorical, num_categories)

    # Combine numerical and categorical attributes
    combined_dataset = combine_attributes(num_rows, numerical_data, categorical_data)

    # Prompt for impurities parameters
    missing_percentage_input = input("Enter the percentage of missing values (0-100) or leave blank for default (5%): ")
    missing_percentage = int(missing_percentage_input) if missing_percentage_input else 5
    num_outliers_input = input("Enter the number of outliers: ")
    num_outliers = int(num_outliers_input) if num_outliers_input else int(round(num_rows * 0.02))
    noise_std_dev_input = input("Enter the standard deviation for noise: ")
    noise_std_dev = float(noise_std_dev_input) if noise_std_dev_input else 0.5

    # Add impurities
    training_dataset = add_impurities(combined_dataset, missing_percentage, num_outliers, noise_std_dev)
    test_dataset = combined_dataset.copy()
    return training_dataset, test_dataset


def prompt_user_input():
    choice = input("Generate dataset from scratch (1) or upload dataset (2)? ")
    if choice == '1':
        choice = input("Generate random correlated dataset (1) or enter values manually (2)? ")
        if choice == '1':
            training_dataset, test_dataset = generate_training_from_scratch(autogen = True)
        else:
            training_dataset, test_dataset = generate_training_from_scratch(autogen = False)
        training_dataset.to_csv('training_dataset.csv', index=True)
        test_dataset.to_csv('test_dataset.csv', index=False)
    elif choice == '2':
        test_dataset = pd.read_csv(input("Enter the path to the dataset: "))
        training_dataset = generate_training_from_uploaded(test_dataset)
    else:
        print("Invalid choice. Please try again.")
        prompt_user_input()
    print("Generated training dataset:")
    print(training_dataset.head())
    training_dataset.to_csv('training_dataset.csv', index=True)


# Call the function to prompt user input and generate the dataset
prompt_user_input()
