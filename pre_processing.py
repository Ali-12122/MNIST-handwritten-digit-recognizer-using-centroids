import numpy as np
import pandas as pd
from math import sqrt


def calculate_image_centroids(image):
    image_as_arr = np.asarray(image)
    flat_image_as_arr = image_as_arr.flatten('C')

    # collapses the image into a unidimensional array
    # for easier manipulation.

    image_dimensions = np.asarray(image_as_arr.shape)
    rows = image_dimensions[0]
    cols = image_dimensions[1]

    rows_position_matrix = np.full(image_as_arr.shape, 0)

    for row in range(0, rows_position_matrix.shape[0], 1):
        rows_position_matrix[row] = np.full(cols, row)

    # This array is an array that a matrix will be built upon which is called
    # the rows position matrix, as it contains the values of the rows for
    # each cell, inside said cell, i.e. each cell in each row contains
    # the number of the row, this is as such as to compute the centroids of the image
    # the matrix is initialized as a matrix of the same dimensions as the image, filled with zeros.
    #
    # The matrix is then iterated upon, row by row, to change the value of each cell of each row
    # to be the number of the row.

    cols_position_matrix = np.full(image_as_arr.shape, 0)
    cols_position_matrix = cols_position_matrix.transpose()

    for column in range(0, cols_position_matrix.shape[0], 1):
        cols_position_matrix[column] = np.full(rows, column)

    # Same idea with the rows_position_matrix, but with the cols.

    cols_position_matrix = cols_position_matrix.transpose()
    # Creating the cols_position_matrix in this way makes the code easier to read and look at,
    # as it creates the matrix in a row major order, but this creates a cols_position_matrix
    # that is transposed, and hence, getting its transpose returns it to its original shape.

    sum_matrix_values = int(np.sum(flat_image_as_arr))

    matrix_values_times_row_values = image_as_arr * rows_position_matrix
    matrix_values_times_column_values = image_as_arr * cols_position_matrix

    flat_matrix_values_times_row_values = matrix_values_times_row_values.flatten('C')
    flat_matrix_values_times_column_values = matrix_values_times_column_values.flatten('C')

    sum_matrix_values_times_row_values = int(np.sum(flat_matrix_values_times_row_values))
    sum_matrix_values_times_column_values = int(np.sum(flat_matrix_values_times_column_values))

    column_centroid = sum_matrix_values_times_column_values / sum_matrix_values
    row_centroid = sum_matrix_values_times_row_values / sum_matrix_values

    return row_centroid, column_centroid


def is_perfect_square(num):
    sqrt_of_num = int(sqrt(num))
    return (sqrt_of_num * sqrt_of_num) == num


def extract_feature_vector(image, num_sub_images):
    # Function to extract the feature vector of the image,
    # the function divides the images int 16 sub_images, computes the centroids of each sub_image
    # and appends them to a feature vector.

    # makes sure the number of sum images is a perfect square for so that the function works properly.
    if not (is_perfect_square(num_sub_images)):
        return str(num_sub_images + " is not a perfect square.")

    else:
        # num_pixels_after_slice_across_row and num_pixels_after_slice_across_columns is the number of
        # slices that would be
        # required to be applied vertically and horizontally to the image to achieve
        # the desired number of subImages.
        image_as_arr = np.asarray(image)
        num_pixels_after_slice_across_row = int(image_as_arr.shape[0] // sqrt(num_sub_images))
        num_pixels_after_slice_across_columns = int(image_as_arr.shape[1] // sqrt(num_sub_images))

        feature_vector = np.full((num_sub_images * 2), 0)

        i = 0
        # Nested for loop to iterate over the image.
        for x in range(0, image_as_arr.shape[0], num_pixels_after_slice_across_row):
            for y in range(0, image_as_arr.shape[1], num_pixels_after_slice_across_columns):
                new_sub_image = image_as_arr[x:x + num_pixels_after_slice_across_row,
                                             y:y + num_pixels_after_slice_across_columns]

                # Computing the centroids and adding them to the feature vector.
                image_centroids = calculate_image_centroids(new_sub_image)
                feature_vector[i] = image_centroids[0]
                feature_vector[i + 1] = image_centroids[1]
                i += 1

    return feature_vector.flatten()


def preprocess_dataset(dataset: pd.DataFrame):
    # Facade function to execute the preprocessing
    # drops the label column, reshapes the each row to resemble a grayscale image
    # and applies extracts the feature vector from each.

    # Dropping the labels.
    dataset_labels = pd.DataFrame(dataset['label'], columns=['label'])
    dataset = dataset.drop('label', axis=1)

    # Creating a matrix to store the feature vectors.
    dataset_feature_vectors = np.full((42000, 32), 0)

    # Applying the extract_feature_vector function on each
    # row after reshaping it to an image, then adding it to
    # the dataset_feature_vectors matrix
    for i in dataset.index:
        current_image = np.asarray(dataset.loc[i])
        current_image = np.reshape(current_image, (28, 28))
        feature_vector = extract_feature_vector(current_image, 16)
        dataset_feature_vectors[i] = feature_vector

    # Creating a label vector for naming the features in the pandas Dataframe returned by this function.
    column_labels = np.full(32, "a")
    for i in range(0, 32, 2):
        x_c = "Xc" + str(i + 1)  # Xc centroid
        y_c = "Yc" + str(i + 1)  # Yc centroid
        column_labels[i] = x_c
        column_labels[i + 1] = y_c

    # Creating the pandas.Dataframe containing the dataset, using the name vector to name the columns,
    # and concatenating the labels(Y, or the state of nature) to the dataframe.
    preprocessed_dataset = pd.DataFrame(data=dataset_feature_vectors, columns=column_labels)
    preprocessed_dataset = pd.concat((preprocessed_dataset, dataset_labels), axis=1)

    return preprocessed_dataset
