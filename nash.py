import numpy as np

def row_maximin(payoff_matrix):
    """
    Finds the row maximin strategy for Player 1 (Row Player).
    :param payoff_matrix: A 2D numpy array representing the game's payoff matrix for player 1.
    :return: The row index corresponding to the maximin strategy and the maximin value.
    """
    # For each row, find the minimum payoff and then return the row that maximizes the minimum payoff
    row_min = np.min(payoff_matrix, axis=1)  # Minimum payoff for each row
    row_maximin_idx = np.argmax(row_min)     # Index of the row with the maximum minimum payoff
    row_maximin_value = row_min[row_maximin_idx]  # The maximum of the minimum values
    return row_maximin_idx, row_maximin_value

def column_minimax(payoff_matrix):
    """
    Finds the column minimax strategy for Player 2 (Column Player).
    :param payoff_matrix: A 2D numpy array representing the game's payoff matrix for player 1.
    :return: The column index corresponding to the minimax strategy and the minimax value.
    """
    # For each column, find the maximum payoff and then return the column that minimizes the maximum payoff
    col_max = np.max(payoff_matrix, axis=0)  # Maximum payoff for each column
    col_minimax_idx = np.argmin(col_max)     # Index of the column with the minimum of the maximum payoffs
    col_minimax_value = col_max[col_minimax_idx]  # The minimum of the maximum values
    return col_minimax_idx, col_minimax_value

def row_and_column_elimination(payoff_matrix):
    """
    Applies row and column elimination to reduce the payoff matrix to 2x2.
    :param payoff_matrix: A 2D numpy array representing the game's payoff matrix.
    :return: A reduced 2x2 payoff matrix.
    """
    while payoff_matrix.shape[0] > 2 and payoff_matrix.shape[1] > 2:
        # Row elimination: eliminate strictly dominated rows
        row_min = np.min(payoff_matrix, axis=1)
        dominated_rows = [i for i in range(len(row_min)) if all(payoff_matrix[i, j] <= row_min[i] for j in range(payoff_matrix.shape[1]))]
        payoff_matrix = np.delete(payoff_matrix, dominated_rows, axis=0)

        # Column elimination: eliminate strictly dominated columns
        col_max = np.max(payoff_matrix, axis=0)
        dominated_cols = [j for j in range(len(col_max)) if all(payoff_matrix[i, j] <= col_max[j] for i in range(payoff_matrix.shape[0]))]
        payoff_matrix = np.delete(payoff_matrix, dominated_cols, axis=1)

        # Ensure that the matrix still has at least 2 rows and columns
        if payoff_matrix.shape[0] <= 2 and payoff_matrix.shape[1] <= 2:
            break

    return payoff_matrix

def find_saddle_point(payoff_matrix):
    """
    Finds the saddle point using maximin, minimax, and row/column elimination.
    :param payoff_matrix: A 2D numpy array representing the game's payoff matrix.
    :return: The row strategy, column strategy, and the value of the game (saddle point).
    """
    # Step 1: Apply row and column elimination to reduce to a 2x2 matrix
    reduced_matrix = row_and_column_elimination(payoff_matrix)

    # Check if we have a 2x2 matrix after elimination
    if reduced_matrix.shape[0] < 2 or reduced_matrix.shape[1] < 2:
        return None  # Matrix is too small to find a saddle point

    # Step 2: Apply row maximin strategy for Player 1
    row_idx, row_maximin_value = row_maximin(reduced_matrix)
    
    # Step 3: Apply column minimax strategy for Player 2
    col_idx, col_minimax_value = column_minimax(reduced_matrix)

    # Step 4: Check if maximin = minimax for the saddle point
    if row_maximin_value == col_minimax_value:
        return row_idx, col_idx, row_maximin_value
    else:
        return None  # No saddle point found

# Example Usage
if __name__ == "__main__":
    # Example payoff matrix for player 1 (row player)
    # payoff_matrix = np.array([
    #     [1, -1, 3],
    #     [2, 0, -2],
    #     [-1, 4, -1]
    # ])

    payoff_matrix = np.array([
    [1, 2],
    [0, 3]
    ])



    result = find_saddle_point(payoff_matrix)

    if result is not None:
        row_idx, col_idx, saddle_value = result
        print(f"Saddle Point Found! Row Player's Strategy: Row {row_idx}, Column Player's Strategy: Column {col_idx}")
        print(f"Value of the Game (Saddle Point): {saddle_value}")
    else:
        print("No Saddle Point Found.")
