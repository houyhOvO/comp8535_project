import csv

# Read the txt file
with open('../dataset/boston.txt', 'r') as txtfile:
    # Skip introducing lines
    for _ in range(22):   # The first 22 lines are the introduction
        txtfile.readline()

    data = txtfile.readlines()

with open('../dataset/boston.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    # Add a header
    header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    writer.writerow(header)

    # Write the csv file
    for i in range(0, len(data), 2):  # The step is 2 because every two lines in the txt file contains a set of data
        # Get a set of data
        row1 = data[i].strip().split(' ')
        row2 = data[i + 1].strip().split(' ')

        # Since the separation of data in the txt file is not uniform,
        # empty strings in row1 and row2 need to be removed.
        cleaned_row1 = [data for data in row1 if data != '']
        cleaned_row2 = [data for data in row2 if data != '']

        # Combine into one row
        row = cleaned_row1 + cleaned_row2
        writer.writerow(row)
