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
    # Regex to catch: model, models, modeling, modeled, etc. (case-insensitive)
    # \b ensures we don't accidentally strip words like "remodel"
    pattern = re.compile(r'\bmodel[s|ing|ed]*\b', re.IGNORECASE)
    # Replace with empty string and clean up double spaces
    cleaned = pattern.sub("", text)
    return " ".join(cleaned.split())

# --- 2. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    # Search both the title and the original abstract (we use the scrubbed one for vectors only)
    full_text = f"{row['label']} {row['original_abstract']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 3. FETCH & BUILD ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            print(f"⚠️ arXiv error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")

if __name__ == "__main__":
    if os.path.exists("docs"): shutil.rmtree("docs")
    os.makedirs("docs", True)

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
            scrubbed = scrub_model_words(f"{r.title}. {r.summary}")
            data_list.append({
                "label": r.title,                 # Hover Title
                "text": scrubbed,                 # Scrubbed content for Topic Modeling
                "original_abstract": r.summary,   # For UI display / metadata
                "url": r.pdf_url,
                "id": r.entry_id.split('/')[-1]
            })
            
        df = pd.DataFrame(data_list)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df.to_parquet(DB_PATH, index=False)
        
        print(f"🧠 Building Map (Scrubbed {len(df)} papers)...")
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # Post-build: Ensure UI uses 'label'
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f: conf = json.load(f)
            conf["name_column"] = "label"
            with open(config_path, "w") as f: json.dump(conf, f)

        # Inject UI Menu
        index_file = "docs/index.html"
        if os.path.exists(index_file):
            overlay = '<div id="lee-menu" style="position:fixed; top:20px; left:20px; z-index:999999;"><button onclick="var t=document.getElementById(\'lee-tab\'); t.style.display=t.style.display===\'none\'?\'block\':\'none\'" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button><div id="lee-tab" style="display:none; margin-top:10px; width:250px; background:#111827; color:white; padding:15px; border-radius:10px; font-family:sans-serif; border:1px solid #374151;"><h3>AI Research Map</h3><p style="font-size:12px;">By Lee Fischman</p></div></div>'
            with open(index_file, "r") as f: content = f.read()
            with open(index_file, "w") as f: f.write(content.replace("<body>", "<body>" + overlay))

        print("✨ Deployment Complete!")
