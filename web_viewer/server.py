#!/usr/bin/env python3
"""
GAN-HTR Web Viewer Server

Serves the standalone HTR viewer and provides API integration
with the GAN document restoration pipeline.

Features:
- Serves static web viewer files
- API endpoint /api/results to list available results
- API endpoint /api/latest to get latest result
- Auto-refresh support for live updates

Usage:
    python server.py [--port 7863]
"""

import http.server
import socketserver
import os
import sys
import argparse
import json
import webbrowser
from pathlib import Path
from functools import partial
from urllib.parse import urlparse, parse_qs
import glob

# Server settings
DEFAULT_PORT = 7863
VIEWER_DIR = Path(__file__).parent
RESULTS_DIR = VIEWER_DIR / "results"


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with CORS support and API endpoints"""
    
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # Parse URL
        parsed = urlparse(self.path)
        
        # API routes
        if parsed.path == '/api/results':
            self.handle_api_results()
        elif parsed.path == '/api/latest':
            self.handle_api_latest()
        else:
            # Serve static files
            super().do_GET()
    
    def handle_api_results(self):
        """Return list of all available results"""
        try:
            results = []
            
            if RESULTS_DIR.exists():
                for doc_dir in sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if doc_dir.is_dir():
                        doc_id = doc_dir.name
                        
                        # Check for image and XML
                        img_files = list(doc_dir.glob("*.jpg")) + list(doc_dir.glob("*.png"))
                        xml_files = list((doc_dir / "page").glob("*.xml")) if (doc_dir / "page").exists() else []
                        
                        if img_files and xml_files:
                            results.append({
                                "id": doc_id,
                                "name": doc_id,
                                "image": f"results/{doc_id}/{img_files[0].name}",
                                "xml": f"results/{doc_id}/page/{xml_files[0].name}",
                                "timestamp": int(doc_dir.stat().st_mtime)
                            })
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"results": results, "count": len(results)}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def handle_api_latest(self):
        """Return the latest result only"""
        try:
            latest = None
            latest_time = 0
            
            if RESULTS_DIR.exists():
                for doc_dir in RESULTS_DIR.iterdir():
                    if doc_dir.is_dir():
                        mtime = doc_dir.stat().st_mtime
                        if mtime > latest_time:
                            doc_id = doc_dir.name
                            img_files = list(doc_dir.glob("*.jpg")) + list(doc_dir.glob("*.png"))
                            xml_files = list((doc_dir / "page").glob("*.xml")) if (doc_dir / "page").exists() else []
                            
                            if img_files and xml_files:
                                latest_time = mtime
                                latest = {
                                    "id": doc_id,
                                    "name": doc_id,
                                    "image": f"results/{doc_id}/{img_files[0].name}",
                                    "xml": f"results/{doc_id}/page/{xml_files[0].name}",
                                    "timestamp": int(mtime)
                                }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"latest": latest}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        # Custom logging (suppress for API calls)
        if '/api/' not in args[0]:
            print(f"[Viewer] {args[0]} - {args[1]}")


def run_server(port: int = DEFAULT_PORT, open_browser: bool = True):
    """Run the web viewer server"""
    
    os.chdir(VIEWER_DIR)
    
    handler = partial(CustomHandler, directory=str(VIEWER_DIR))
    
    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        url = f"http://localhost:{port}"
        print("=" * 60)
        print("  GAN-HTR Document Viewer Server")
        print("=" * 60)
        print(f"  📂 Serving from: {VIEWER_DIR}")
        print(f"  🌐 URL: {url}")
        print(f"  📡 Network: http://0.0.0.0:{port}")
        print("=" * 60)
        print("\n  Press Ctrl+C to stop the server\n")
        
        if open_browser:
            try:
                webbrowser.open(url)
            except:
                pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n  Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="GAN-HTR Web Viewer Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, 
                       help=f"Port to run server (default: {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser automatically")
    
    args = parser.parse_args()
    run_server(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
