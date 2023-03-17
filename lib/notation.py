import re

A = ord('a')

# Assumes dimensions of 8 or fewer
REGEX = re.compile(r'\d[a-h]')

def max_col(dim):
    return chr(A + dim - 1)

def coords2notation(coords):
    r, c = coords
    return f"{r+1}{chr(A + c)}"

def notation2coords(pos):
    r, c = pos
    return (int(r) - 1, ord(c) - A)

def extract_notation(message):
    '''
    Search for first occurrence of Othello notation and return coordinates
    '''
    match = REGEX.search(message)
    if match is None:
        return None
    return notation2coords(match.group(0))

