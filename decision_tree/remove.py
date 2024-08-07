import os

print(os.path.abspath('.'))

for i in os.listdir("data/testDigits"):
    if i.startswith("."):
        os.remove(f"data/testDigits/{i}")

for i in os.listdir("data/trainingDigits"):
    if i.startswith("."):
        os.remove(f"data/trainingDigits/{i}")
        print(f"data/trainingDigits/{i}")
