import os
import sys
import re

# regex to extract the parsingg time number
#pattern = re.compile(r"Total parsingg time:\s*([0-9.]+)\s*seconds")
pattern = re.compile(r"Total parsing time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds\s*\(with output\)")

parssing_time_dict = {}
t = sys.argv[1]
print(os.listdir(t))
# get all directory names in the current directory
for d in os.listdir(t):
    d = os.path.join(t, d)
    if not os.path.isdir(d):
        continue
    log_path = os.path.join(d, "log_test.txt")
    print(log_path)
    if not os.path.isfile(log_path):
        continue
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                parssing_time_dict[d] = float(match.group(1))
                break

# assert the dict size is exactly 14
assert len(parssing_time_dict
           ) == 14, f"Expected 14 entries, got {len(parssing_time_dict)}"

# compute and print the average (1 digit after decimal point)
average_time = sum(parssing_time_dict.values()) / len(parssing_time_dict)
print(f"{average_time:.1f}")
