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
    def translate_path(self, path):
        # optional overlay: --models DIR serves DIR at /models/ (stand-in networks for tests)
        if MODELS and path.split('?')[0].startswith('/models/'):
            return os.path.join(MODELS, path.split('?')[0][len('/models/'):])
        return super().translate_path(path)
    def log_message(self, *a): pass
args = [a for a in sys.argv[1:] if not a.startswith('--')]
MODELS = sys.argv[sys.argv.index('--models') + 1] if '--models' in sys.argv else None
if MODELS: args.remove(MODELS)
root = args[0]; port = int(args[1]) if len(args) > 1 else 8765
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(H, directory=root)).serve_forever()
