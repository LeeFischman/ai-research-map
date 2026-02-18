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

# --- 1. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    full_text = f"{row['label']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 2. ARXIV FETCHER ---
def fetch_results_with_retry(client, search, max_retries=5):
    for i in range(max_retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            print(f"⚠️ arXiv error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("Max retries exceeded")

# --- 3. THE KILLSWITCH ---
def apply_universal_fix(docs_path):
    config_path = os.path.join(docs_path, "data", "config.json")
    if not os.path.exists(config_path): return

    with open(config_path, "r") as f:
        conf = json.load(f)

    # Force the UI to use our 'label' column for EVERY display element
    conf["name_column"] = "label"
    conf["label_column"] = "label"
    
    # Disable automated topic labeling by pointing it to a static column
    if "topic_label_column" in conf or "cluster_labels" in conf:
        conf["topic_label_column"] = "Reputation"
    
    # Ensure column mappings are explicit
    conf["column_mappings"] = {
        "label": "label",
        "text": "text",
        "Reputation": "Reputation"
    }

    with open(config_path, "w") as f:
        json.dump(conf, f, indent=4)

    # Fix the UI Overlay
    index_file = os.path.join(docs_path, "index.html")
    if os.path.exists(index_file):
        ui_blob = """
        <div id="lee-overlay" style="position:fixed; top:20px; left:20px; z-index:999999;">
            <button id="info-toggle" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold;">⚙️ Menu</button>
            <div id="info-tab" style="display:none; margin-top:10px; width:280px; background:#111827; border:1px solid #374151; color:white; padding:20px; border-radius:12px; font-family:sans-serif; box-shadow:0 10px 15px rgba(0,0,0,0.5);">
                <h2 style="margin:0; color:#60a5fa; font-size:18px;">AI Research Map</h2>
                <p style="font-size:13px;">Created by <a href="https://www.linkedin.com/in/lee-fischman/" target="_blank" style="color:#3b82f6;">Lee Fischman</a></p>
                <hr style="border:0; border-top:1px solid #374151; margin:10px 0;">
                <p style="font-size:12px;">Color by <b>'Reputation'</b> in the side menu.</p>
            </div>
        </div>
        <script>
            document.getElementById('info-toggle').onclick = function() {
                var tab = document.getElementById('info-tab');
                tab.style.display = (tab.style.display === 'none' || tab.style.display === '') ? 'block' : 'none';
            };
        </script>
        """
        with open(index_file, "r") as f: content = f.read()
        with open(index_file, "w") as f: f.write(content.replace("<body>", "<body>" + ui_blob))

# --- 4. EXECUTION ---
if __name__ == "__main__":
    if os.path.exists("docs"): shutil.rmtree("docs")
    os.makedirs("docs", exist_ok=True)

    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=10, num_retries=10)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = fetch_results_with_retry(client, search)
    if results:
        df = pd.DataFrame([{
            "label": r.title,                 # DISPLAY TITLE
            "text": f"{r.title}. {r.summary}", # VECTOR CONTENT
            "url": r.pdf_url,
            "id": r.entry_id.split('/')[-1]
        } for r in results])

        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        df.to_parquet(DB_PATH, index=False)
        
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        apply_universal_fix("docs")
        print("✨ Sync Complete!")
