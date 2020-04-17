import numpy as np
import re
import argparse


def check_if_valid_line(line):
    """
    Checks if a given line is a valid sudoku line
    :param line: a line of characters
    :return: True if a line is a valid sudoku line, false otherwise
    """
    return bool(re.match("^[0-9.]{81}$", line))


def convert_to_matrix(line):
    """
    Converts a line of text into a 9x9 matrix, where . is replaced into 0
    :param line: a line of text
    :return: a 9x9 numpy array representing a 9x9 sudoku board
    """
    return np.reshape(list(map(int, line.replace('.', '0'))), (9, 9))


def get_empty_squares(board):
    """
    Returns a vector with coordinates of empty squares left in the board
    :param board: a 9x9 sudoku board
    :return: a vector with coordinates of empty squares
    """
    return np.asarray(np.where(board == 0)).T


def get_non_empty_squares(board):
    """
    Returns a vector with coordinates of non-empty squares in the board
    :param board: a 9x9 sudoku board
    :return: a vector with coordinates of non-empty squares
    """
    return np.asarray(np.where(board != 0)).T


def convert_to_string(board):
    """
    Converts a 9x9 sudoku board to a string
    :param board: a 9x9 sudoku board
    :return: a string representing the 9x9 sudoku board
    """
    return ''.join(map(str, board.flatten()))


def check_column(column, value, board):
    """
    Checks if a value exists in a column of the board
    :param column: number of column
    :param value: value to be checked
    :param board: a 9x9 sudoku board
    :return: True if value exists, False otherwise
    """
    if value in board[:, column]:
        return True
    return False


def check_row(row, value, board):
    """
    Checks if a value exists in a row of the board
    :param row: number of row
    :param value: value to be checked
    :param board: a 9x9 sudoku board
    :return: True if value exists, False otherwise
    """
    if value in board[row, :]:
        return True
    return False


def check_square(row, column, value, board):
    """
    Checks if a value exists in a 3x3 square of the board
    :param row: number of row
    :param column: number of column
    :param value: value to be checked
    :param board: a 9x9 sudoku board
    :return: True if value exists, False otherwise
    """
    if value in board[row // 3 * 3:int(row / 3) * 3 + 3, column // 3 * 3:int(column / 3) * 3 + 3]:
        return True
    return False


def check(row, column, value, board):
    """
    Checks if a value can be inserted into a board at a specific position
    :param row: number of row
    :param column: number of column
    :param value: value to be checked
    :param board: a 9x9 sudoku board
    :return: True if value can be inserted, False otherwise
    """
    return not (check_column(column, value, board) or check_row(row, value, board) or check_square(row, column, value,
                                                                                                   board))


def solve_backtracking(board):
    """
    Solves a sudoku board using a backtracking algorithm
    :param board: a 9x9 sudoku board
    :return: True if the sudoku board can be solved, False otherwise
    """
    global backtracks
    domain = list(range(1, 10))
    empty_squares = get_empty_squares(board)

    if len(empty_squares) == 0:
        return True

    row, col = empty_squares[0][0], empty_squares[0][1]

    while domain:
        value = domain.pop(0)

        if check(row, col, value, board):
            board[row][col] = value
            if solve_backtracking(board):
                return True
            else:
                backtracks += 1
                board[row][col] = 0

    return False


def solve_forward_checking(board):
    """
    Solves a sudoku board using a backtracking algorithm combined with forward-checking
    :param board: a 9x9 sudoku board
    :return: True if the sudoku board can be solved, False otherwise
    """
    global backtracks
    empty_squares = get_empty_squares(board)

    if len(empty_squares) == 0:
        return True

    row, col = empty_squares[0][0], empty_squares[0][1]

    remaining_values = get_remaining_values(board)
    values = remaining_values[row][col]

    while values:
        value = values.pop()

        if forward_check(remaining_values, row, col, value):
            board[row][col] = value
            if solve_forward_checking(board):
                return True
            else:
                backtracks += 1
                board[row][col] = 0

    return False


def get_remaining_values(board):
    """
    Generated a domain array for the sudoku board
    :param board: a 9x9 sudoku board
    :return: a 9x9 numpy array of sets representing domains for specific squares
    """
    remaining_values = np.array([set(range(1, 10)) for _ in range(81)]).reshape(9, 9)
    non_empty_squares = get_non_empty_squares(board)
    for square in non_empty_squares:
        row = square[0]
        col = square[1]
        value = board[row][col]

        # remove from column
        for domain in remaining_values[:, col]:
            domain.discard(value)

        # remove from row
        for domain in remaining_values[row, :]:
            domain.discard(value)

        # remove from 3x3 square
        temp = remaining_values[row // 3 * 3:int(row / 3) * 3 + 3, col // 3 * 3:int(col / 3) * 3 + 3].flatten()
        for domain in temp:
            domain.discard(value)

        remaining_values[row][col] = {value}
    return remaining_values


def forward_check(remaining_values, row, col, value):
    """
    Checks if a value inserted at a specific position violates rules of the sudoku board
    :param remaining_values: a domain array
    :param row: number of row
    :param col: number of column
    :param value: value to be checked
    :return: True if insertion of the value does not violate rules of sudoku, False otherwise
    """
    # check column
    for i in range(9):
        if i == row:
            continue

        x = remaining_values[row][i]
        if len(x) == 1 and list(x)[0] == value:
            return False

    # check row
    for i in range(9):
        if i == col:
            continue

        x = remaining_values[i][col]
        if len(x) == 1 and list(x)[0] == value:
            return False

    # check 3x3 square
    block_row = row // 3
    block_col = col // 3
    for i in range(3):
        for j in range(3):
            if [block_row * 3 + i, block_col * 3 + j] == [row, col]:
                continue

            x = remaining_values[block_row * 3 + i][block_col * 3 + j]
            if len(x) == 1 and list(x)[0] == value:
                return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudoku solver using backtracking and forward-checking algorithms")
    parser.add_argument("sudoku", help="a line of text representing sudoku board, 81 characters with . representing an "
                                      "empty square, example:\n"
                                      "......2.6...7....8.5......36.1.5.32.....37.....2......9...7......36..4915..81....")
    parser.add_argument("-fc", "--fc", help="enables forward-checking", action="store_true")
    args = parser.parse_args()
    text = args.sudoku
    if check_if_valid_line(text):
        backtracks = 0
        solvable = False
        sudoku = convert_to_matrix(text)
        if args.fc:
            solvable = solve_forward_checking(sudoku)
        else:
            solvable = solve_backtracking(sudoku)
        if solvable:
            print(convert_to_string(sudoku), "\n", backtracks, "backtracks")
        else:
            print("nonsolvable")
    else:
        print("Incorrect format of sudoku, read --help for details")
