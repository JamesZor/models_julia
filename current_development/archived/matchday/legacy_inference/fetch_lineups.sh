#!/usr/bin/env bash

# Directory to save lineups
SAVE_DIR="data/lineups"
mkdir -p "$SAVE_DIR"

# Check if arguments are provided
if [ "$#" -lt 1 ]; then
    # If stdin is a TTY, print usage and exit
    if [ -t 0 ]; then
        echo "Usage: $0 <match_id1> [match_id2] ..."
        echo "Or read from stdin: echo \"15238092 15238093\" | $0"
        exit 1
    fi
fi

# Read match IDs from arguments or stdin
MATCH_IDS=()
if [ "$#" -gt 0 ]; then
    MATCH_IDS=("$@")
else
    # Read from stdin
    while read -r line; do
        for id in $line; do
            MATCH_IDS+=("$id")
        done
    done
fi

USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SUCCESS_COUNT=0

for mid in "${MATCH_IDS[@]}"; do
    # Strip non-digit characters
    mid=$(echo "$mid" | tr -cd '0-9')
    if [ -z "$mid" ]; then
        continue
    fi
    
    echo -n "Fetching lineup for match ID: $mid..."
    URL="https://api.sofascore.com/api/v1/event/${mid}/lineups"
    OUT_FILE="${SAVE_DIR}/${mid}.json"
    
    HTTP_CODE=$(curl -s -o "$OUT_FILE" -w "%{http_code}" -H "User-Agent: $USER_AGENT" "$URL")
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        # Check if response is empty or null or missing players
        if [ ! -s "$OUT_FILE" ] || grep -q '^null$' "$OUT_FILE" || ! grep -q '"players"' "$OUT_FILE"; then
            echo " [!] Lineup not available yet or null/invalid for match $mid."
            rm -f "$OUT_FILE"
        else
            echo " [+] Saved to $OUT_FILE"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    else
        echo " [x] Failed (HTTP Code: $HTTP_CODE)"
        rm -f "$OUT_FILE"
    fi
    
    # Sleep to avoid rate limiting
    sleep 1
done

echo -e "\nDone! Successfully fetched and saved $SUCCESS_COUNT lineups."
