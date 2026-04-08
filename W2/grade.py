import sys

if __name__ == "__main__":
    grades = [int(x) for x in sys.argv[1:]]
    mean = sum(grades) / len(grades)
    if mean >= 5:
        print(f"{mean} Pass")
    else:
        print(f"{mean} Fail")