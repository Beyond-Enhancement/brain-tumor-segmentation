#!/usr/bin/env python3
"""Static server for development/tests: loopback only, cross-origin isolation headers
(needed for multi-threaded WebAssembly), correct MIME types. Usage: serve.py ROOT [PORT]"""
import http.server, sys, os, functools
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.mjs': 'text/javascript', '.js': 'text/javascript',
                      '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.json': 'application/json', '.gz': 'application/gzip'}
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
    def do_GET(self):
        # optional: --addons DIR emulates the launcher's add-on model listing for tests
        p = self.path.split('?')[0]
        if ADDONS and p in ('/__bts/addons', '/__bts/addons/'):
            import json
            items = [{'name': f, 'size': os.path.getsize(os.path.join(ADDONS, f))} for f in sorted(os.listdir(ADDONS)) if f.lower().endswith('.btsmodel')]
            b = json.dumps(items).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b); return
        if ADDONS and p.startswith('/__bts/addons/'):
            name = p[len('/__bts/addons/'):]
            f = os.path.join(ADDONS, name)
            if '/' in name or not name.lower().endswith('.btsmodel') or not os.path.isfile(f):
                self.send_error(404); return
            b = open(f, 'rb').read()
            self.send_response(200); self.send_header('Content-Type', 'application/zip'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b); return
        return super().do_GET()
    def translate_path(self, path):
        # optional overlay: --models DIR serves DIR at /models/ (stand-in networks for tests)
        if MODELS and path.split('?')[0].startswith('/models/'):
            return os.path.join(MODELS, path.split('?')[0][len('/models/'):])
        return super().translate_path(path)
    def log_message(self, *a): pass
args = [a for a in sys.argv[1:] if not a.startswith('--')]
MODELS = sys.argv[sys.argv.index('--models') + 1] if '--models' in sys.argv else None
if MODELS: args.remove(MODELS)
ADDONS = sys.argv[sys.argv.index('--addons') + 1] if '--addons' in sys.argv else None
if ADDONS: args.remove(ADDONS)
root = args[0]; port = int(args[1]) if len(args) > 1 else 8765
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(H, directory=root)).serve_forever()
