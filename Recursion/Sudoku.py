def solveSudoku(board):
    solvePartialSudoku(0, 0, board)
    return board
    
def solvePartialSudoku(row, col, board):
    if col == len(board[row]):
        col = 0
        row += 1
        if row == len(board):
            return True

    if board[row][col] == 0:
        return tryDigitAt(row, col, board)

    return solvePartialSudoku(row, col+1, board)

def tryDigitAt(row, col, board):
    for digit in range(1, 10):
        if isValidAtPosition(digit, row, col, board):
            board[row][col] = digit
            if solvePartialSudoku(row, col+1, board):
                return True

    board[row][col] = 0
    return False
    
def isValidAtPosition(value, row, col, board):
    rowIsValid = value not in board[row]
    colIsValid = value not in map(lambda r:r[col], board)
    '''colIsValid = True
    for i in range(len(board)):
        if board[i][col] == value:
            colIsValid = False'''

    if not rowIsValid or not colIsValid:
        return False

    subgridRow = (row // 3)*3
    subgridCol = (col // 3)*3
    for rowIdx in range(3):
        for colIdx in range(3):
            rowToCheck = subgridRow + rowIdx
            colToCheck = subgridCol + colIdx

            if board[rowToCheck][colToCheck] == value:
                return False

    return True
