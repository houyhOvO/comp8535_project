def update_model_results(model_name, mse, r_squared):
    """
    Updates the model results
    :param model_name: model name
    :param mse: Mean Squared Error of a model
    :param r_squared: R-squared of a model
    """
    try:
        with open('../models/results.txt', 'r') as file:
            lines = file.readlines()
            results = {line.split(':')[0]: line.split(':')[1].strip() for line in lines}
    except FileNotFoundError:
        results = {}

    # Update the results of the specified model
    results[model_name] = f"MSE={mse}, R-squared={r_squared}"

    with open('../models/results.txt', 'w') as file:
        for key, value in results.items():
            file.write(f"{key}:{value}\n")
