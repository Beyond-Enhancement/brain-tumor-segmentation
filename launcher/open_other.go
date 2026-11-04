//go:build !windows && !darwin

package main

import (
	"fmt"
	"os/exec"
	"syscall"
)

func openAppWindow(url string) error {
	for _, b := range []string{"google-chrome", "google-chrome-stable", "microsoft-edge", "chromium", "chromium-browser", "brave-browser"} {
		if p, err := exec.LookPath(b); err == nil {
			cmd := exec.Command(p, "--app="+url)
			cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
			if err := cmd.Start(); err == nil {
				go cmd.Wait()
				return nil
			}
		}
	}
	cmd := exec.Command("xdg-open", url)
	if err := cmd.Start(); err != nil {
		return err
	}
	go cmd.Wait()
	return nil
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
