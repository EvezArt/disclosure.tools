"""disclosure.tools web server — UAP Eigenforensics"""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json, time

app = FastAPI(title="disclosure.tools", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "service": "disclosure.tools", "version": "1.0.0", "ts": int(time.time())}

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static_index.html")

@app.post("/api/v1/analyze")
async def analyze_documents(file: UploadFile = File(...)):
    """Upload a JSON corpus, get eigenforensics analysis."""
    content = await file.read()
    try:
        documents = json.loads(content)
        if not isinstance(documents, list):
            documents = [documents]
    except:
        return {"error": "Invalid JSON. Send array of documents."}
    
    # Import and run spectral engine
    import sys
    sys.path.insert(0, ".")
    from gap_detector.spectral_engine import SpectralEngine
    engine = SpectralEngine(operator="evez666")
    engine.load_corpus(documents)
    result = engine.analyze()
    return result

@app.get("/api/v1/demo")
def demo_analysis():
    """Run eigenforensics on the built-in AARO sample corpus."""
    import sys
    sys.path.insert(0, ".")
    from gap_detector.spectral_engine import SpectralEngine
    
    sample_docs = [
        {"id": "AARO-2024-001", "text": "Unidentified anomalous phenomena assessment office report", "date": "2024-03"},
        {"id": "AARO-2024-002", "text": "UAP incident report redacted Naval aviation encounter", "date": "2024-03"},
        {"id": "AARO-2024-003", "text": "FOIA release AARO historical records review", "date": "2024-06"},
        {"id": "AARO-2024-004", "text": "Department of Defense statement on UAP classification", "date": "2024-06"},
        {"id": "AARO-2024-005", "text": "National Archives UAP document release partial", "date": "2024-09"},
        {"id": "AARO-2024-006", "text": "Pentagon UAP task force historical analysis volumes 1-9", "date": "2024-09"},
    ]
    engine = SpectralEngine(operator="evez666")
    engine.load_corpus(sample_docs)
    return engine.analyze()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8087)
