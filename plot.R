data <- read.csv("data/train.csv")

y <- data["target"]

print(table(y))
