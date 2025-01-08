#!/usr/bin/env bash
set -euo pipefail

# 1) Run black in check mode
if black --check .; then
  STATUS="passing"
  COLOR="brightgreen"
else
  STATUS="failing"
  COLOR="red"
fi

# 2) Construct the badge link
# For example: "![Black](https://img.shields.io/badge/Black-passing-brightgreen)"
BADGE="![Black](https://img.shields.io/badge/Black-${STATUS}-${COLOR})"

# 3) Insert or replace in README
SEARCH='!\[Black\]\(https://img\.shields\.io/badge/Black-(passing|failing)-(brightgreen|red)\)'
if grep -Eq "$SEARCH" README.md; then
  sed -i -E "s#${SEARCH}#${BADGE}#g" README.md
else
  echo -e "\n$BADGE" >> README.md
fi

git add README.md
