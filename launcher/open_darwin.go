//go:build darwin

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// openAppWindow opens the URL as a chromeless app window in Chrome, Edge, Brave or
// Chromium when one is installed, otherwise in the default browser (e.g. Safari).
func openAppWindow(url string) error {
	home, _ := os.UserHomeDir()
	apps := []string{"Google Chrome.app", "Microsoft Edge.app", "Brave Browser.app", "Chromium.app"}
	for _, app := range apps {
		for _, dir := range []string{"/Applications", filepath.Join(home, "Applications")} {
			p := filepath.Join(dir, app)
			if st, err := os.Stat(p); err == nil && st.IsDir() {
				// -n lets the --app argument reach an already-running browser.
				if err := exec.Command("open", "-na", p, "--args", "--app="+url).Run(); err == nil {
					return nil
				}
			}
		}
	}
	return exec.Command("open", url).Run()
}

func notify(title, msg string, isErr bool) {
	fmt.Println()
	if isErr {
		fmt.Println("ERROR:", msg)
	} else {
		fmt.Println(msg)
	}
	fmt.Println()
}
