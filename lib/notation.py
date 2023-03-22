import re

A = ord('a')

# Assumes dimensions of 8 or fewer
REGEX = re.compile(r'[a-h]\d')

def max_col(dim):
    return chr(A + dim - 1)

def coords2notation(coords):
    r, c = coords
    return f"{chr(A + c)}{r+1}"

def notation2coords(pos):
    c, r = pos
    return (int(r) - 1, ord(c) - A)

def extract_notation(message):
    '''
    Search for first occurrence of Othello notation and return coordinates
    '''
    match = REGEX.search(message)
    if match is None:
        return None
    return notation2coords(match.group(0))

def extract_all_notation(message):
    '''
    Return all Othello notation in message translated into coordinates
    '''
    return [notation2coords(m) for m in REGEX.findall(message)]
