import sys

def listsum(arr):
    return sum(arr)

def deduplicate(arr: list) -> list:
    return list(dict.fromkeys(arr))

def sorttuples(arr):
    return sorted(arr, key=lambda x: x[1])


def squarecubes(arr):
    squares = [x**2 for x in arr]
    cubes = [x**3 for x in arr]
    return squares, cubes


def grade_checl(arg_arr):
    grades = [int(x) for x in sys.argv[1:]]
    mean = sum(grades) / len(grades)
    if mean >= 5:
        print("Pass")
    else:
        print("Fail")


if __name__ == "__main__":
    assert listsum([1, 2, 3, 4]) == 10
    assert deduplicate([1, 2, 3, 2, 4, 1, 5]) == [1, 2, 3, 4, 5]
    print(sys.argv[1:])