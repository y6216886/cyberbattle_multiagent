#!/usr/bin/env bash

REQ_FILE="requirements.txt"

if [[ ! -f "$REQ_FILE" ]]; then
    echo "Error: $REQ_FILE not found!"
    exit 1
fi

SUCCESS=()
FAILED=()

while IFS= read -r pkg || [[ -n "$pkg" ]]; do
    # skip empty lines and comments
    [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue

    echo "--------------------------------------------------"
    echo "Installing: $pkg"
    echo "--------------------------------------------------"

    pip install "$pkg"
    if [[ $? -eq 0 ]]; then
        echo "[OK] $pkg installed successfully."
        SUCCESS+=("$pkg")
    else
        echo "[FAILED] $pkg installation failed."
        FAILED+=("$pkg")
    fi
done < "$REQ_FILE"

echo
echo "============== Installation Summary =============="
echo "Successful installs (${#SUCCESS[@]}):"
printf '  %s\n' "${SUCCESS[@]}"

echo
echo "Failed installs (${#FAILED[@]}):"
printf '  %s\n' "${FAILED[@]}"

echo "=================================================="

