import arxiv
import pandas as pd
import subprocess
import os
from datetime import datetime

def fetch_arxiv_data(limit=1000):
    # Initialize the client correctly for v2.x
    client = arxiv.Client()
    
    # Define the search parameters
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    data = []
    # FIX: Pass the 'search' object into client.results()
    for r in client.results(search):
        # We add 'date' and 'primary_category' as extra columns for filtering
        data.append({
            "title": r.title,
            "text_for_embedding": f"{r.title}. {r.summary}",
            "url": r.pdf_url,
            "date": r.published.strftime("%Y-%m-%d"),
            "category": r.primary_category,
            "summary_short": r.summary[:200] + "..." # For quick hover info
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    print(f"📥 Starting update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = fetch_arxiv_data(1000)
    df.to_parquet("papers.parquet")
    
    print("🧠 Building Atlas with SPECTER2...")
    subprocess.run([
        "embedding-atlas", "papers.parquet",
        "--text", "text_for_embedding",
        "--model", "allenai/specter2_base",
        "--export-application", "site.zip"
    ], check=True)
    
    # Create docs folder and unzip
    os.makedirs("docs", exist_ok=True)
    os.system("unzip -o site.zip -d docs/")

    # FIX: Tell GitHub not to use Jekyll
    with open("docs/.nojekyll", "w") as f:
        f.write("")
        
    # NEW: Create a small JSON file for the 'Last Updated' badge on the README
    with open("docs/status.json", "w") as f:
        f.write(f'{{"last_update": "{datetime.now().strftime("%Y-%m-%d")}"}}')
        
    print("✅ Success! Map updated.")

