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
    # Remove "model" so the vector engine focuses on content
    pattern = re.compile(r'\bmodel[s|ing|ed]*\b', re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return " ".join(cleaned.split())

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
    full_text = f"{row['title']} {row['original_abstract']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std"

# --- 3. FETCH & PREPARE ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            time.sleep(wait)
    raise Exception("Max retries exceeded")

if __name__ == "__main__":
    clear_docs_contents("docs")

    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=10)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = list(client.results(search))
    if results:
        data_list = []
        for r in results:
            data_list.append({
                "title": r.title,                 # Standard name for UI
                "original_abstract": r.summary,   # Keep for metadata
                "text": scrub_model_words(f"{r.title}. {r.summary}"), # Vectors
                "url": r.pdf_url,
                "id": r.entry_id.split('/')[-1]
            })
            
        df = pd.DataFrame(data_list)
        
        # --- THE TROJAN HORSE STRATEGY ---
        # 1. Create the Reputation column
        df['group'] = df.apply(calculate_reputation, axis=1)
        
        # 2. Duplicate it as 'topic' to force the tool to use it for cluster names
        # This replaces "Model" with "Reputation Enhanced"
        df['topic'] = df['group']
        
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map (Forced Grouping)...")
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # --- FINAL CONFIG OVERRIDE ---
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                conf = json.load(f)
            
            # 1. Force Title Display
            conf["name_column"] = "title"
            conf["label_column"] = "title"
            
            # 2. Force Grouping by Reputation
            # By setting 'topic_label_column' to 'group', we ensure the big text 
            # over the clusters is your Reputation data, not "Model".
            conf["topic_label_column"] = "group"
            conf["color_by"] = "group"
            
            # 3. Ensure Columns Exist in Browser Map
            if "column_mappings" not in conf:
                conf["column_mappings"] = {}
            conf["column_mappings"]["title"] = "title"
            conf["column_mappings"]["group"] = "group"
            conf["column_mappings"]["url"] = "url"
            
            with open(config_path, "w") as f:
                json.dump(conf, f, indent=4)
            print("✅ Config locked: Title=Label, Group=Reputation.")

        # --- UI MENU ---
        index_file = "docs/index.html"
        if os.path.exists(index_file):
            overlay = '<div id="lee-menu" style="position:fixed; top:20px; left:20px; z-index:999999;"><button onclick="var t=document.getElementById(\'lee-tab\'); t.style.display=t.style.display===\'none\'?\'block\':\'none\'" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button><div id="lee-tab" style="display:none; margin-top:10px; width:250px; background:#111827; color:white; padding:15px; border-radius:10px; font-family:sans-serif; border:1px solid #374151;"><h3>AI Research Map</h3><p style="font-size:12px;">By Lee Fischman</p></div></div>'
            with open(index_file, "r") as f: 
                content = f.read()
            with open(index_file, "w") as f: 
                f.write(content.replace("<body>", "<body>" + overlay))

        print("✨ Sync Complete!")
