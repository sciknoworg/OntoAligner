#!/usr/bin/env bash

set -e

if [ ! -f pyproject.toml ]; then
    echo "pyproject.toml not found!"
    exit 1
fi

echo "Generating clean requirements.txt with direct dependencies..."

> requirements.txt

awk '
/^\[tool\.poetry\.dependencies\]/ {section="deps"; next}
/^\[tool\.poetry\.dev-dependencies\]/ {section="dev"; next}
/^\[/ {section=""; next}

section!="" && NF>0 {
    if ($1 ~ /^python/) next

    gsub(/"/, "", $0)
    gsub(/ /, "", $0)

    split($0, a, "=")
    pkg=a[1]
    ver=a[2]

    # Leave hyphens as-is (do not convert to underscores)

    if(ver=="" || ver=="*") {
        print pkg
    } else {
        # Remove ^ or ~ for pip-friendly pin
        gsub(/\^|~/, "", ver)
        # Add == only if not already present
        if(index(ver, "==")==0) {
            print pkg "==" ver
        } else {
            print pkg ver
        }
    }
}' pyproject.toml > requirements.txt

echo "Done! requirements.txt created."
