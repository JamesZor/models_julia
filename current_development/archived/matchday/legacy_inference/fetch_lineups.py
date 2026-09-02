#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
import urllib.error
import json

def fetch_lineup(match_id, save_dir):
    url = f"https://api.sofascore.com/api/v1/event/{match_id}/lineups"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{match_id}.json")
    
    req = urllib.request.Request(url, headers=headers)
    
    print(f"Fetching lineups for match {match_id}...", end="", flush=True)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = response.read().decode('utf-8')
                parsed = json.loads(data)
                if not parsed or parsed == "null":
                    print(" [!] Empty/null lineup returned.")
                    return False
                
                # Check if it has the required structure (home/away players)
                if "home" not in parsed or "players" not in parsed["home"]:
                    print(" [!] Lineups not released yet (missing players field).")
                    return False
                
                # Write to file
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2)
                
                print(f" [+] Saved to {filepath}")
                return True
            else:
                print(f" [x] Failed (Status: {response.status})")
                return False
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(" [x] Not found (404). Lineups might not be available yet.")
        else:
            print(f" [x] HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f" [x] Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 fetch_lineups.py <match_id1> [match_id2] ...")
        print("Or pass match IDs via stdin: echo \"15238092\" | python3 fetch_lineups.py")
        
        # Check if stdin has data
        if sys.stdin.isatty():
            sys.exit(1)
        
        # Read from stdin
        match_ids = sys.stdin.read().split()
    else:
        match_ids = sys.argv[1:]
        
    save_dir = "data/lineups"
    
    success_count = 0
    for mid in match_ids:
        mid = mid.strip()
        if not mid.isdigit():
            continue
        
        success = fetch_lineup(int(mid), save_dir)
        if success:
            success_count += 1
        time.sleep(1.0) # Rate limit courtesy
        
    print(f"\nDone! Successfully fetched and saved {success_count} lineups.")

if __name__ == "__main__":
    main()
