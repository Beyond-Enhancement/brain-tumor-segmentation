//go:build windows

package main

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"unsafe"
)

// openAppWindow opens the URL as a chromeless app window in Edge (present on every
// Windows 10/11 machine) or Chrome, falling back to the default browser.
func openAppWindow(url string) error {
	var candidates []string
	for _, base := range []string{os.Getenv("ProgramFiles(x86)"), os.Getenv("ProgramFiles"), os.Getenv("LocalAppData")} {
		if base == "" {
			continue
		}
		candidates = append(candidates,
			filepath.Join(base, `Microsoft\Edge\Application\msedge.exe`),
		)
	}
	for _, base := range []string{os.Getenv("ProgramFiles"), os.Getenv("ProgramFiles(x86)"), os.Getenv("LocalAppData")} {
		if base == "" {
			continue
		}
		candidates = append(candidates,
			filepath.Join(base, `Google\Chrome\Application\chrome.exe`),
		)
	}
	for _, exe := range candidates {
		if st, err := os.Stat(exe); err == nil && !st.IsDir() {
			cmd := exec.Command(exe, "--app="+url, "--start-maximized")
			if err := cmd.Start(); err == nil {
				go cmd.Wait()
				return nil
			}
		}
	}
	cmd := exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	if err := cmd.Start(); err != nil {
		return errors.New("no browser could be started")
	}
	go cmd.Wait()
	return nil
}

var (
	user32      = syscall.NewLazyDLL("user32.dll")
	messageBoxW = user32.NewProc("MessageBoxW")
)

// notify shows a message box (this build has no console window).
func notify(title, msg string, isErr bool) {
	t, _ := syscall.UTF16PtrFromString(title)
	m, _ := syscall.UTF16PtrFromString(msg)
	const (
		mbIconError = 0x10
		mbIconInfo  = 0x40
		mbTopmost   = 0x40000
		mbSetFg     = 0x10000
	)
	flags := uintptr(mbIconInfo | mbTopmost | mbSetFg)
	if isErr {
		flags = uintptr(mbIconError | mbTopmost | mbSetFg)
	}
	messageBoxW.Call(0, uintptr(unsafe.Pointer(m)), uintptr(unsafe.Pointer(t)), flags)
}
