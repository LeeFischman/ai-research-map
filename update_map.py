import arxiv
import pandas as pd
import subprocess
import os
import re
import json
import time
import shutil
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"

# --- 1. THE SCRUBBER ---
def scrub_model_words(text):
    # Strip model, models, modeling, etc. to force better cluster names
    pattern = re.compile(r'\bmodel[s|ing|ed]*\b', re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return " ".join(cleaned.split())

# --- 2. THE DIRECTORY CLEANER (Permission-Safe) ---
def clear_docs_contents(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# --- 3. REPUTATION & FETCH ---
def calculate_reputation(row):
    score = 0
    full_text = f"{row['label']} {row['original_abstract']}".lower()
    if any(inst in full_text for inst in ["mit", "stanford", "deepmind", "openai", "meta ai"]): score += 3
    if "github.com" in full_text: score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            time.sleep(wait)
    raise Exception("Max retries exceeded")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # CLEAR CONTENTS ONLY to avoid rsync permission issues
    clear_docs_contents("docs")

    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=10)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = fetch_results_with_retry(client, search)
    if results:
        data_list = []
        for r in results:
            # The 'text' column gets the scrubbed version for the vector/topic engine
            scrubbed = scrub_model_words(f"{r.title}. {r.summary}")
            data_list.append({
                "label": r.title,
                "text": scrubbed,
                "original_abstract": r.summary,
                "url": r.pdf_url,
                "id": r.entry_id.split('/')[-1]
            })
            
        df = pd.DataFrame(data_list)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df.to_parquet(DB_PATH, index=False)
        
        print(f"🧠 Building Map...")
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        # Unzip into the existing docs folder
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # Post-build UI Config
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f: conf = json.load(f)
            conf["name_column"] = "label"
            with open(config_path, "w") as f: json.dump(conf, f)

        # UI Overlay
        index_file = "docs/index.html"
        if os.path.exists(index_file):
            overlay = '<div id="lee-menu" style="position:fixed; top:20px; left:20px; z-index:999999;"><button onclick="var t=document.getElementById(\'lee-tab\'); t.style.display=t.style.display===\'none\'?\'block\':\'none\'" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button><div id="lee-tab" style="display:none; margin-top:10px; width:250px; background:#111827; color:white; padding:15px; border-radius:10px; font-family:sans-serif; border:1px solid #374151;"><h3>AI Research Map</h3><p style="font-size:12px;">By Lee Fischman</p></div></div>'
            with open(index_file, "r") as f: content = f.read()
            with open(index_file, "w") as f: f.write(content.replace("<body>", "<body>" + overlay))

        print("✨ Process Successful!")
