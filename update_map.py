import arxiv
import pandas as pd
import subprocess
import os
import gc
import re
import json
import time
import shutil
from datetime import datetime, timedelta, timezone

DB_PATH = "database.parquet"

# --- 1. REPUTATION LOGIC ---
INSTITUTION_PATTERN = re.compile(r"\b(" + "|".join([
    "MIT", "Stanford", "CMU", "UC Berkeley", "Harvard", "Princeton", "Cornell",
    "DeepMind", "OpenAI", "Anthropic", "FAIR", "Meta AI", "Microsoft Research"
]) + r")\b", re.IGNORECASE)

def calculate_reputation(row):
    score = 0
    # Search both the vector text and the display label
    full_text = f"{row['label']} {row['text']}".lower()
    if INSTITUTION_PATTERN.search(full_text): score += 3
    authors = row.get('authors', [])
    if len(authors) >= 5: score += 1
    if any(k in full_text for k in ['github.com', 'huggingface.co']): score += 2
    return "Reputation Enhanced" if score >= 4 else "Reputation Std."

# --- 2. THE PATIENT FETCHER (arXiv 429 Protection) ---
def fetch_with_backoff(client, search, retries=5):
    for i in range(retries):
        try:
            return list(client.results(search))
        except Exception as e:
            wait = (i + 1) * 60
            print(f"⚠️ arXiv error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise Exception("Failed to fetch from arXiv.")

# --- 3. UI OVERLAY ---
def inject_ui(docs_path):
    index_file = os.path.join(docs_path, "index.html")
    if os.path.exists(index_file):
        ui_blob = """
        <div id="custom-overlay" style="position:fixed; top:20px; left:20px; z-index:999999;">
            <button id="info-toggle" style="background:#2563eb; color:white; border:none; padding:10px 15px; border-radius:8px; cursor:pointer; font-weight:bold; box-shadow:0 4px 6px rgba(0,0,0,0.3);">⚙️ Menu</button>
            <div id="info-tab" style="display:none; margin-top:10px; width:260px; background:#111827; border:1px solid #374151; color:white; padding:20px; border-radius:12px; font-family:sans-serif; box-shadow:0 10px 15px rgba(0,0,0,0.5);">
                <h2 style="margin:0; color:#60a5fa; font-size:18px;">AI Research Map</h2>
                <p style="font-size:13px; color:#9ca3af;">By Lee Fischman</p>
                <hr style="border:0; border-top:1px solid #374151; margin:10px 0;">
                <p style="font-size:12px;">💡 Color by <b>'Reputation'</b> in the Atlas side-menu to see lab-weighted clusters.</p>
            </div>
        </div>
        <script>
            document.getElementById('info-toggle').onclick = function() {
                var tab = document.getElementById('info-tab');
                tab.style.display = (tab.style.display === 'none' || tab.style.display === '') ? 'block' : 'none';
            };
        </script>
        """
        with open(index_file, "r") as f:
            content = f.read()
        with open(index_file, "w") as f:
            f.write(content.replace("<body>", "<body>" + ui_blob))

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure a fresh slate for the docs directory
    if os.path.exists("docs"):
        shutil.rmtree("docs")
    os.makedirs("docs", exist_ok=True)

    now = datetime.now(timezone.utc)
    client = arxiv.Client(page_size=100, delay_seconds=10, num_retries=10)
    search = arxiv.Search(
        query=f"cat:cs.AI AND submittedDate:[{(now-timedelta(days=5)).strftime('%Y%m%d%H%M')} TO {now.strftime('%Y%m%d%H%M')}]", 
        max_results=250
    )
    
    results = fetch_with_backoff(client, search)
    if results:
        # DUAL COLUMN STRATEGY
        df = pd.DataFrame([{
            "id": r.entry_id.split('/')[-1],
            "label": r.title,                  # Target for UI Display
            "text": f"{r.title}. {r.summary}",  # Target for Vector Embeddings
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d"),
            "authors": [a.name for a in r.authors]
        } for r in results])
        
        df = df.drop_duplicates(subset='id').reset_index(drop=True)
        df['Reputation'] = df.apply(calculate_reputation, axis=1)
        
        # Save clean Parquet
        df.to_parquet(DB_PATH, index=False)
        
        print("🧠 Building Map (Dual-Column Strategy)...")
        # We target 'text' for the model, while 'label' remains for the UI
        subprocess.run([
            "embedding-atlas", DB_PATH, 
            "--text", "text", 
            "--model", "allenai/specter2_base", 
            "--export-application", "site.zip"
        ], check=True)
        
        # Deploy application files
        os.system("unzip -o site.zip -d docs/ && touch docs/.nojekyll")
        
        # Post-processing: Explicitly tell config to use our label column
        config_path = "docs/data/config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                conf = json.load(f)
            conf["name_column"] = "label"
            with open(config_path, "w") as f:
                json.dump(conf, f)

        inject_ui("docs")
        print("✨ Deployment Successful!")
    else:
        print("No papers found.")
