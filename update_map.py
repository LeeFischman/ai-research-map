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
STOPWORDS_PATH = "stop_words.csv"

# --- 1. THE DICTIONARY SCRUBBER ---
def load_stop_words():
    if os.path.exists(STOPWORDS_PATH):
        sw_df = pd.read_csv(STOPWORDS_PATH)
        return set(sw_df['word'].str.lower().tolist())
    return {"model", "models", "modeling"}

STOP_WORDS = load_stop_words()

def scrub_text(text):
    words = text.split()
    cleaned = [w for w in words if w.lower().strip('.,()[]{}') not in STOP_WORDS]
    return " ".join(cleaned)

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
            print(f"Skipped {file_path}: {e}")

# --- 2. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    full_text = f"{row['title']} {row['summary']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std"

# --- 3. REINFORCED FETCH (429 Protection) ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "Too Many Requests" in err_msg:
                # Heavy backoff for 429s: 2min, 4min, 8min...
                wait = (2 ** i) * 120 
                print(f"🛑 arXiv Rate Limit (429). Cooling down for {wait}s...")
                time.sleep(wait)
            else:
                wait = 30
                print(f"⚠️ Connection error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise Exception("Max retries exceeded. arXiv is blocking this IP.")

if __name__ == "__main__":
    clear_docs_contents("docs")

    # Increased delay_seconds to 5 to avoid triggering the 429 in the first place
    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=10)
    
    now = datetime.now(timezone.utc)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = fetch_results_with_retry(client, search)
    if results:
        data_list = []
        for r in results:
            data_list.append({
                "title": r.title,
                "summary": r.summary,
                "text": scrub_text(f"{r.title}. {r.summary}"), 
                "url": r.pdf_url,
                "id": r.entry_id.split('/')[-1],
                "topic": "AI Research Feed" # Forced floating label
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
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # --- CONFIG FIX ---
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                conf = json.load(f)
            
            conf["name_column"] = "title"
            conf["label_column"] = "title"
            
            if "column_mappings" not in conf:
                conf["column_mappings"] = {}
            conf["column_mappings"]["title"] = "title"
            conf["column_mappings"]["Reputation"] = "Reputation"
            conf["column_mappings"]["topic"] = "topic"
            conf["column_mappings"]["url"] = "url"
            
            conf["topic_label_column"] = "topic"
            conf["color_by"] = "Reputation"
            
            with open(config_path, "w") as f:
                json.dump(conf, f, indent=4)

        # --- UI MENU ---
        index_file = "docs/index.html"
        if os.path.exists(index_file):
            overlay = '<div id="lee-menu" style="position:fixed; top:20px; left:20px; z-index:999999;"><button onclick="var t=document.getElementById(\'lee-tab\'); t.style.display=t.style.display===\'none\'?\'block\':\'none\'" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button><div id="lee-tab" style="display:none; margin-top:10px; width:250px; background:#111827; color:white; padding:15px; border-radius:10px; font-family:sans-serif; border:1px solid #374151;"><h3>AI Research Map</h3><p style="font-size:12px;">By Lee Fischman</p></div></div>'
            with open(index_file, "r") as f: content = f.read()
            with open(index_file, "w") as f: f.write(content.replace("<body>", "<body>" + overlay))

        print("✨ Deployment Successful!")
