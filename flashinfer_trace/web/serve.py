#!/usr/bin/env python3
"""Simple HTTP server to serve the Quant-GEMM Bench web UI."""
import http.server
import socketserver
import os
import json
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs
import cgi

PORT = 8888

# Directory containing definitions
DEFINITIONS_DIR = Path(__file__).parent.parent / "definitions"

def scan_kernels():
    """Scan definitions directory for all kernel JSON files."""
    kernels = []

    if not DEFINITIONS_DIR.exists():
        return kernels

    for category_dir in DEFINITIONS_DIR.iterdir():
        if category_dir.is_dir():
            for json_file in category_dir.glob("*.json"):
                # Create relative path from web/ directory
                rel_path = f"../definitions/{category_dir.name}/{json_file.name}"
                kernels.append({
                    "name": json_file.stem,
                    "path": rel_path,
                    "category": category_dir.name
                })

    # Sort by name
    kernels.sort(key=lambda x: x["name"])
    return kernels

def get_model_info():
    """Load model architectures info."""
    models_file = DEFINITIONS_DIR / "model_architectures.json"
    if models_file.exists():
        with open(models_file, 'r') as f:
            return json.load(f)
    return {"models": {}, "op_categories": {}}

def get_op_categories():
    """Get unique op categories from all kernels."""
    model_info = get_model_info()
    return model_info.get("op_categories", {})

class APIRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves static files and API endpoints."""

    def do_GET(self):
        # API endpoints
        if self.path == '/api/kernels':
            self.send_json_response(scan_kernels())
        elif self.path == '/api/models':
            self.send_json_response(get_model_info())
        elif self.path == '/api/op-categories':
            self.send_json_response(get_op_categories())
        else:
            # Serve static files
            super().do_GET()

    def send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def main():
    # Change to the flashinfer_trace directory (parent of web)
    # This allows access to both web/ and definitions/
    flashinfer_dir = Path(__file__).parent.parent
    os.chdir(flashinfer_dir)

    # Scan kernels on startup
    kernels = scan_kernels()
    print(f"\n🚀 Quant-GEMM Bench is running!")
    print(f"   Open: http://localhost:{PORT}/web/")
    print(f"\n📋 Found {len(kernels)} kernel definitions:")
    for kernel in kernels:
        cat_marker = {
            'quant_gemm': '⚡',
            'quant_vec_dot': '🔵',
            'quantize': '📦'
        }.get(kernel['category'], '📄')
        print(f"   {cat_marker} {kernel['name']}")
    print(f"\n   Press Ctrl+C to stop\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), APIRequestHandler) as httpd:
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}/web/')
        except:
            pass

        httpd.serve_forever()

if __name__ == "__main__":
    main()
