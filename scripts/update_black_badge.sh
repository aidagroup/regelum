#!/usr/bin/env bash
set -eu

# 1) Run black in check mode
if black --check .; then
  STATUS="passing"
  COLOR="brightgreen"
else
  STATUS="failing"
  COLOR="red"
fi

# 2) Construct the badge link
BADGE="![Black](https://img.shields.io/badge/Black-${STATUS}-${COLOR})"
echo "Status: $STATUS, Color: $COLOR"

# 3) Replace or insert black badge in README
SEARCH='!\[Black\]\(https://img\.shields\.io/badge/Black-[a-z]*-[a-z]*\)'
REPLACE="![Black](https://img.shields.io/badge/Black-${STATUS}-${COLOR})"

if grep -Eq "$SEARCH" README.md; then
  sed -i -E "s|$SEARCH|$REPLACE|g" README.md
else
  echo -e "\n$REPLACE" >> README.md
fi

# 4) Stage the updated README
git add README.md
