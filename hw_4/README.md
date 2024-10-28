# Go Setup Guide

This guide will help you install Go, set up your environment, and run a Go program.

## Prerequisites

- A system with a supported operating system (Windows, macOS, Linux)
- Access to a terminal/command line interface
- Internet connection for downloading Go

## Step 1: Download Go

Visit the official Go website and download the latest version of Go for your operating system:

- [Go Downloads](https://go.dev/dl/)

Choose the appropriate installer based on your operating system:

- **Windows**: Download the `.msi` installer.
- **macOS**: Download the `.pkg` installer.
- **Linux**: Download the `.tar.gz` archive and install it using the terminal.

## Step 2: Install Go

### On Linux
1. Open the terminal and navigate to the directory where the `.tar.gz` file was downloaded.
2. Extract the archive using:

    ```bash
    tar -C /usr/local -xzf go1.x.x.linux-amd64.tar.gz
    ```

   (Replace `go1.x.x.linux-amd64.tar.gz` with the actual filename you downloaded.)

3. Add Go to your PATH by editing your shell profile (e.g., `~/.bashrc`, `~/.zshrc`, or `~/.profile`):

    ```bash
    export PATH=$PATH:/usr/local/go/bin
    ```

4. Reload the shell or run:

    ```bash
    source ~/.bashrc  # or source ~/.zshrc, depending on your shell
    ```

5. Verify the installation by running:

    ```bash
    go version
    ```

### On Windows
1. Run the downloaded `.msi` file.
2. Follow the installation instructions.
3. After installation, restart your terminal or command prompt.

### On macOS
1. Open the `.pkg` file and follow the prompts to install.
2. Once the installation is complete, verify the installation in the terminal by running:

    ```bash
    go version
    ```

## Step 3: Set Up Your Go Workspace

To write and run Go programs, you'll need to set up a workspace:

1. Create a directory for your Go projects (optional but recommended):

    ```bash
    mkdir ~/go-workspace
    cd ~/go-workspace
    ```

2. Inside this directory, create a folder for your project, e.g., `hello-world`:

    ```bash
    mkdir hello-world
    cd hello-world
    ```

## Step 4: Write a Go Program

Create a Go file (e.g., `main.go`):

```bash
touch main.go
```

Open it in a text editor and add the following sample code:

```
package main

import "fmt"

func main() {
    fmt.Println("Hello, Go!")
}
```

Step 5: Run Your Go Program
To run your Go program, use the go run command:

```
go run main.go
```
