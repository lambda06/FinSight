"""
server.py — FastAPI backend server for FinSight Web Dashboard.

Serves the API and static assets:
  - GET  /api/users   → return available user IDs and names
  - POST /api/query   → process natural language queries
  - Static files      → serve UI files from static/ and Matplotlib output charts from output/
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tabular_rag_pipeline.pipeline import TransactionRAGPipeline
from tabular_rag_pipeline.exceptions import UserNotFoundError

# Create directories if they do not exist
os.makedirs("static", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Initialise global RAG pipeline
print("Loading FinSight Tabular RAG Pipeline...")
pipeline = TransactionRAGPipeline()
print("Pipeline initialized successfully.")

app = FastAPI(title="FinSight API", version="1.0.0")

# Request model for query endpoint
class QueryRequest(BaseModel):
    user_id: str
    query: str

# ── API Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/users")
def get_users():
    """Retrieve all available users in the database and their names."""
    try:
        user_ids = pipeline.store.get_all_user_ids()
        users = []
        for uid in user_ids:
            try:
                name = pipeline.store.get_user_name(uid)
                # Fetch profile details for basic dashboard metadata
                profile = pipeline.cache.get_profile(uid)
                if not profile:
                    profile = pipeline.store.compute_user_profile(uid)
                    pipeline.cache.set_profile(uid, profile)
                
                users.append({
                    "id": uid,
                    "name": name,
                    "date_range": f"{profile['date_range']['start']} to {profile['date_range']['end']}",
                    "transactions": profile["total_transactions"],
                    "avg_spend": profile["avg_monthly_expense"]
                })
            except Exception:
                # If profile computation fails for a user, skip details but return ID/name
                users.append({
                    "id": uid,
                    "name": uid,
                    "date_range": "N/A",
                    "transactions": 0,
                    "avg_spend": 0.0
                })
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def run_query(req: QueryRequest):
    """Run a query through the 10-stage RAG pipeline."""
    if not req.user_id or not req.query:
        raise HTTPException(status_code=400, detail="Missing user_id or query")
    
    try:
        result = pipeline.query(req.user_id, req.query)
        # Convert absolute output chart paths to relative web paths
        web_visualizations = []
        for path in result.get("visualizations", []):
            filename = os.path.basename(path)
            # The output directory is mounted as /output static folder
            web_visualizations.append(f"/output/{filename}")
        
        result["visualizations"] = web_visualizations
        return result
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Static File Mounts ─────────────────────────────────────────────────────────

# Mount output folder to serve generated PNG charts directly in the browser
app.mount("/output", StaticFiles(directory="output"), name="output")

# Mount static folder for frontend UI assets
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    """Serve the index.html page as the main app entrypoint."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "running", "message": "static/index.html not found. Please create it."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
