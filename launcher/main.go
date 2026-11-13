// Brain Tumor Segmentation Workbench - standalone launcher.
//
// The whole web app (page, viewer, ONNX Runtime, HD-BET and HD-GLIO networks, demo case) is
// embedded in this binary. On launch it serves the app on the loopback interface only and opens
// it in a browser app window. Nothing is installed and nothing leaves the computer: all image
// processing runs inside the browser, and the page's Content-Security-Policy stops the browser
// from connecting anywhere except this local server.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed all:site
var siteFS embed.FS

const (
	appName   = "Brain Tumor Segmentation Workbench"
	appVer    = "1.5.0"
	pingPath  = "/__bts/ping"
	eventPath = "/__bts/events"
	addonPath = "/__bts/addons" // add-on model files (*.btsmodel) found next to the executable
	pingReply = "brain-tumor-segmentation-workbench"

	// A fixed port keeps the page origin stable between runs.
	defaultPort = 47821

	idleGrace    = 30 * time.Second // after the last window closes
	firstConnect = 5 * time.Minute  // if no window ever connects
)

// The browser may only load from, and connect to, this server. 'wasm-unsafe-eval' is needed to
// compile WebAssembly (ONNX Runtime, image codecs); blob: lets the page hand files to the user.
const csp = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; " +
	"connect-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; " +
	"object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

var contentTypes = map[string]string{
	".html":  "text/html; charset=utf-8",
	".js":    "text/javascript; charset=utf-8",
	".mjs":   "text/javascript; charset=utf-8",
	".css":   "text/css; charset=utf-8",
	".json":  "application/json",
	".wasm":  "application/wasm",
	".woff2": "font/woff2",
	".gz":    "application/gzip",
	".txt":   "text/plain; charset=utf-8",
	".png":   "image/png",
	".svg":   "image/svg+xml",
	".onnx":  "application/octet-stream",
	".pack":  "application/octet-stream",
}

// ---------------------------------------------------------------------------------------
// liveness: exit once every app window has been closed for a while

type liveness struct {
	mu       sync.Mutex
	clients  int
	ever     bool
	lastSeen time.Time
	started  time.Time
}

func (l *liveness) touch() { l.mu.Lock(); l.lastSeen = time.Now(); l.mu.Unlock() }
func (l *liveness) connect() {
	l.mu.Lock()
	l.clients++
	l.ever = true
	l.lastSeen = time.Now()
	l.mu.Unlock()
}
func (l *liveness) disconnect() { l.mu.Lock(); l.clients--; l.lastSeen = time.Now(); l.mu.Unlock() }
func (l *liveness) busy() bool  { l.mu.Lock(); defer l.mu.Unlock(); return l.clients > 0 }

func (l *liveness) shouldExit() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.clients > 0 {
		return false
	}
	if l.ever {
		return time.Since(l.lastSeen) > idleGrace
	}
	return time.Since(l.started) > firstConnect && time.Since(l.lastSeen) > idleGrace
}

// ---------------------------------------------------------------------------------------

func main() {
	port := flag.Int("port", defaultPort, "loopback port to serve on")
	noBrowser := flag.Bool("no-browser", false, "do not open a browser window")
	stay := flag.Bool("stay", false, "keep running after the last window closes")
	flag.Parse()

	fmt.Printf("%s %s\n", appName, appVer)

	existing := fmt.Sprintf("http://127.0.0.1:%d/", *port)
	if isOurServer(existing) {
		fmt.Println("Already running at", existing)
		if !*noBrowser {
			if err := openAppWindow(existing); err != nil {
				notify(appName, "The workbench is already running.\n\nOpen this address in Edge or Chrome:\n"+existing, false)
			}
		}
		return
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", *port))
	if err != nil {
		ln, err = net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			notify(appName, "Could not start the local server:\n"+err.Error(), true)
			os.Exit(1)
		}
	}
	actualPort := ln.Addr().(*net.TCPAddr).Port
	url := fmt.Sprintf("http://127.0.0.1:%d/", actualPort)

	live := &liveness{started: time.Now(), lastSeen: time.Now()}
	srv := &http.Server{Handler: newHandler(actualPort, live), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			notify(appName, "The local server stopped unexpectedly:\n"+err.Error(), true)
			os.Exit(1)
		}
	}()

	fmt.Println("Serving at", url)
	fmt.Println("Everything runs on this computer; no images or data are uploaded.")
	if !*noBrowser {
		if err := openAppWindow(url); err != nil {
			go notify(appName, "Could not open a browser automatically.\n\nOpen this address in Edge or Chrome:\n"+url, false)
		}
	}
	for {
		time.Sleep(2 * time.Second)
		if !*stay && live.shouldExit() {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			_ = srv.Shutdown(ctx)
			cancel()
			fmt.Println("Workbench window closed; exiting.")
			return
		}
	}
}

func isOurServer(base string) bool {
	client := &http.Client{Timeout: 1500 * time.Millisecond, Transport: &http.Transport{Proxy: nil}}
	resp, err := client.Get(base + strings.TrimPrefix(pingPath, "/"))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
	return resp.StatusCode == 200 && strings.TrimSpace(string(b)) == pingReply
}

func newHandler(port int, live *liveness) http.Handler {
	sub, err := fs.Sub(siteFS, "site")
	if err != nil {
		panic(err)
	}
	// files stored gzip-compressed in the binary (listed by build/prepare_site.py)
	gz := map[string]bool{}
	if b, err := fs.ReadFile(sub, "gzip.json"); err == nil {
		var list []string
		if json.Unmarshal(b, &list) == nil {
			for _, p := range list {
				gz["/"+p] = true
			}
		}
	}
	files := http.FileServer(http.FS(sub))
	allowedHosts := map[string]bool{
		fmt.Sprintf("127.0.0.1:%d", port): true,
		fmt.Sprintf("localhost:%d", port): true,
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Only answer requests addressed to the loopback origin (blocks DNS rebinding).
		if !allowedHosts[r.Host] {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer")
		// cross-origin isolation: enables SharedArrayBuffer for multi-threaded WebAssembly
		h.Set("Cross-Origin-Opener-Policy", "same-origin")
		h.Set("Cross-Origin-Embedder-Policy", "require-corp")
		h.Set("Cross-Origin-Resource-Policy", "same-origin")
		h.Set("Content-Security-Policy", csp)
		h.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
		live.touch()

		switch r.URL.Path {
		case pingPath:
			h.Set("Content-Type", "text/plain; charset=utf-8")
			h.Set("Cache-Control", "no-store")
			io.WriteString(w, pingReply)
			return
		case eventPath:
			serveEvents(w, r, live)
			return
		case addonPath:
			h.Set("Content-Type", "application/json")
			h.Set("Cache-Control", "no-store")
			b, _ := json.Marshal(listAddons())
			w.Write(b)
			return
		}
		if strings.HasPrefix(r.URL.Path, addonPath+"/") {
			serveAddon(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		p := r.URL.Path
		if p == "/" {
			p = "/index.html"
		}
		ext := strings.ToLower(path.Ext(p))
		if ct, ok := contentTypes[ext]; ok {
			h.Set("Content-Type", ct)
		}
		if ext == ".html" {
			h.Set("Cache-Control", "no-cache")
		} else {
			h.Set("Cache-Control", "public, max-age=3600")
		}
		if gz[p] {
			f, err := sub.Open(strings.TrimPrefix(p, "/") + ".gz")
			if err == nil {
				defer f.Close()
				if st, err := f.Stat(); err == nil {
					h.Set("Content-Length", strconv.FormatInt(st.Size(), 10))
				}
				h.Set("Content-Encoding", "gzip")
				h.Set("Vary", "Accept-Encoding")
				if r.Method == http.MethodHead {
					return
				}
				io.Copy(w, f)
				return
			}
		}
		files.ServeHTTP(w, r)
	})
}

// ---------------------------------------------------------------------------------------
// Add-on models: optional extra networks shipped as separate *.btsmodel files (too large to
// embed under GitHub's file-size limit). They are read from the folder that holds this program,
// listed by name only, and served read-only; nothing else on disk is reachable.

type addon struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

func addonDir() string {
	exe, err := os.Executable()
	if err != nil {
		return ""
	}
	if p, err := filepath.EvalSymlinks(exe); err == nil {
		exe = p
	}
	return filepath.Dir(exe)
}

func listAddons() []addon {
	out := []addon{}
	dir := addonDir()
	if dir == "" {
		return out
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return out
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".btsmodel") {
			continue
		}
		if info, err := e.Info(); err == nil && info.Mode().IsRegular() && info.Size() > 0 && info.Size() < 1<<30 {
			out = append(out, addon{Name: e.Name(), Size: info.Size()})
		}
		if len(out) >= 16 {
			break
		}
	}
	return out
}

func serveAddon(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimPrefix(r.URL.Path, addonPath+"/")
	for _, a := range listAddons() {
		if a.Name == name { // exact match with a listed file: no other paths can be requested
			f, err := os.Open(filepath.Join(addonDir(), a.Name))
			if err != nil {
				http.Error(w, "not found", http.StatusNotFound)
				return
			}
			defer f.Close()
			w.Header().Set("Content-Type", "application/zip")
			w.Header().Set("Cache-Control", "no-store")
			http.ServeContent(w, r, a.Name, time.Time{}, f)
			return
		}
	}
	http.Error(w, "not found", http.StatusNotFound)
}

// serveEvents holds a long-lived Server-Sent Events stream open for each app window, so the
// launcher knows when the last window has gone.
func serveEvents(w http.ResponseWriter, r *http.Request, live *liveness) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-store")
	live.connect()
	defer live.disconnect()
	io.WriteString(w, "retry: 3000\n: connected\n\n")
	flusher.Flush()
	t := time.NewTicker(15 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-t.C:
			if _, err := io.WriteString(w, ": keep-alive\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}
