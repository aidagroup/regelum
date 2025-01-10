#!/usr/bin/env bash
set -eu

# 1) Run tests locally with coverage
python -m pytest --cov=regelum --cov-config=.coveragerc . | tee coverage.txt

# 2) Extract coverage percentage from the line containing "TOTAL"
COVERAGE=$(grep "TOTAL" coverage.txt | awk '{print $NF}' | sed 's/%//')
if [ -z "$COVERAGE" ]; then
  COVERAGE="0"
fi

# 3) Pick a color based on thresholds
if [ "$(echo "$COVERAGE >= 90" | bc -l)" -eq 1 ]; then
  COLOR="brightgreen"
elif [ "$(echo "$COVERAGE >= 80" | bc -l)" -eq 1 ]; then
  COLOR="green"
elif [ "$(echo "$COVERAGE >= 70" | bc -l)" -eq 1 ]; then
  COLOR="yellow"
else
  COLOR="red"
fi

echo "Coverage: $COVERAGE%, Color: $COLOR"

# 4) Replace or insert coverage badge in README
SEARCH='!\[Coverage\]\(https://img\.shields\.io/badge/coverage-[0-9.]*%25-[a-z]*\)'
REPLACE="![Coverage](https://img.shields.io/badge/coverage-${COVERAGE}%25-${COLOR})"

# If badge exists, replace it; if not, add a new line with coverage badge
if grep -Eq "$SEARCH" README.md; then
  sed -i -E "s|$SEARCH|$REPLACE|g" README.md
else
  echo -e "\n$REPLACE" >> README.md
fi

# 5) Stage the updated README for commit
git add README.md
