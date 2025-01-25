import re

def parse_log_file(file_path):
    acc_values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'ACC:(\d+\.\d+)', line)
            if match:
                acc_values.append(float(match.group(1)))
    return acc_values

def main():
    file_path = 'tmp_wisconsin_2.log'
    acc_values = parse_log_file(file_path)
    acc_values.sort()
    print(acc_values)

if __name__ == "__main__":
    main()