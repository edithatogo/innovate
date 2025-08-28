#!/bin/bash

# Documentation deployment script for innovate library
# This script builds and optionally serves the documentation locally

set -e  # Exit on any error

echo "🚀 Innovate Documentation Builder"
echo "================================="

# Change to docs directory
cd "$(dirname "$0")"

# Function to display help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build     Build documentation (default)"
    echo "  serve     Build and serve documentation locally"
    echo "  clean     Clean build directory"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build     # Build documentation only"
    echo "  $0 serve     # Build and serve on localhost:8000"
    echo "  $0 clean     # Clean build artifacts"
}

# Function to build documentation
build_docs() {
    echo "📚 Building documentation..."
    
    # Clean previous build
    echo "🧹 Cleaning previous build..."
    make clean
    
    # Check if required packages are installed
    echo "🔍 Checking dependencies..."
    python -c "import sphinx, sphinx_rtd_theme, myst_parser" 2>/dev/null || {
        echo "❌ Missing dependencies. Installing..."
        pip install sphinx sphinx-rtd-theme myst-parser nbsphinx
    }
    
    # Build documentation
    echo "🔨 Building HTML documentation..."
    make dev
    
    if [ $? -eq 0 ]; then
        echo "✅ Documentation built successfully!"
        echo "📁 Output directory: build/html/"
        echo "🌐 Open build/html/index.html in your browser"
    else
        echo "❌ Documentation build failed!"
        exit 1
    fi
}

# Function to serve documentation locally
serve_docs() {
    build_docs
    
    echo "🌐 Starting local server..."
    echo "📡 Documentation will be available at: http://localhost:8000"
    echo "⏹️  Press Ctrl+C to stop the server"
    
    cd build/html
    python -m http.server 8000
}

# Function to clean build artifacts
clean_docs() {
    echo "🧹 Cleaning documentation build..."
    make clean
    echo "✅ Build artifacts cleaned!"
}

# Main script logic
case "${1:-build}" in
    "build")
        build_docs
        ;;
    "serve")
        serve_docs
        ;;
    "clean")
        clean_docs
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac