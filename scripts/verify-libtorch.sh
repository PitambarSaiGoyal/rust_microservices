#!/bin/bash
# Verification script for libtorch installation
# Part of tch-rs migration Phase 1

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "  libtorch Verification Script"
echo "================================================"
echo ""

# Check if LIBTORCH environment variable is set
if [ -z "$LIBTORCH" ]; then
    echo -e "${RED}❌ LIBTORCH environment variable not set${NC}"
    echo ""
    echo "Please set the LIBTORCH environment variable:"
    echo "  export LIBTORCH=/path/to/libtorch"
    echo ""
    echo "See rust/docs/TCH_SETUP.md for detailed instructions."
    exit 1
fi

echo -e "${GREEN}✅ LIBTORCH environment variable set${NC}"
echo "   Path: $LIBTORCH"
echo ""

# Check if the path exists
if [ ! -d "$LIBTORCH" ]; then
    echo -e "${RED}❌ LIBTORCH path does not exist: $LIBTORCH${NC}"
    echo ""
    echo "Please verify the path and ensure libtorch is installed."
    echo "See rust/docs/TCH_SETUP.md for installation instructions."
    exit 1
fi

echo -e "${GREEN}✅ libtorch directory exists${NC}"
echo ""

# Check for required directories
echo "Checking directory structure..."

if [ -d "$LIBTORCH/lib" ]; then
    echo -e "${GREEN}✅ lib directory found${NC}"
else
    echo -e "${RED}❌ lib directory not found${NC}"
    exit 1
fi

if [ -d "$LIBTORCH/include" ]; then
    echo -e "${GREEN}✅ include directory found${NC}"
else
    echo -e "${RED}❌ include directory not found${NC}"
    exit 1
fi

echo ""

# Check for libtorch library
echo "Checking for libtorch libraries..."

LIBTORCH_LIB=""
if [ -f "$LIBTORCH/lib/libtorch.so" ]; then
    LIBTORCH_LIB="$LIBTORCH/lib/libtorch.so"
    echo -e "${GREEN}✅ libtorch.so found (Linux)${NC}"
elif [ -f "$LIBTORCH/lib/libtorch.dylib" ]; then
    LIBTORCH_LIB="$LIBTORCH/lib/libtorch.dylib"
    echo -e "${GREEN}✅ libtorch.dylib found (macOS)${NC}"
elif [ -f "$LIBTORCH/lib/libtorch_cpu.so" ]; then
    LIBTORCH_LIB="$LIBTORCH/lib/libtorch_cpu.so"
    echo -e "${GREEN}✅ libtorch_cpu.so found (Linux CPU)${NC}"
elif [ -f "$LIBTORCH/lib/libtorch_cpu.dylib" ]; then
    LIBTORCH_LIB="$LIBTORCH/lib/libtorch_cpu.dylib"
    echo -e "${GREEN}✅ libtorch_cpu.dylib found (macOS)${NC}"
else
    echo -e "${RED}❌ libtorch library not found${NC}"
    echo "Expected: libtorch.so, libtorch.dylib, libtorch_cpu.so, or libtorch_cpu.dylib"
    exit 1
fi

echo ""

# Check for CUDA support
echo "Checking backend support..."

CUDA_AVAILABLE=false
MPS_AVAILABLE=false
CPU_AVAILABLE=true

if ls "$LIBTORCH/lib/"libcudart* 1> /dev/null 2>&1; then
    CUDA_AVAILABLE=true
    echo -e "${GREEN}✅ CUDA libraries found${NC}"

    # Try to detect CUDA version
    if [ -f "$LIBTORCH/lib/libcudart.so.11.0" ]; then
        echo "   CUDA version: 11.0"
    elif [ -f "$LIBTORCH/lib/libcudart.so.11.8" ]; then
        echo "   CUDA version: 11.8"
    elif [ -f "$LIBTORCH/lib/libcudart.so.12.1" ]; then
        echo "   CUDA version: 12.1"
    else
        echo "   CUDA version: Unknown (library found)"
    fi
else
    echo -e "${YELLOW}⚠️  CUDA libraries not found (CPU-only build)${NC}"
fi

# Check for cuDNN
if ls "$LIBTORCH/lib/"libcudnn* 1> /dev/null 2>&1; then
    echo -e "${GREEN}✅ cuDNN libraries found${NC}"
fi

# Detect macOS for MPS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check macOS version for MPS support (requires 12.0+)
    MACOS_VERSION=$(sw_vers -productVersion | cut -d '.' -f 1)
    if [ "$MACOS_VERSION" -ge 12 ]; then
        # Check if M1/M2/M3 (Apple Silicon)
        if [[ $(uname -m) == "arm64" ]]; then
            MPS_AVAILABLE=true
            echo -e "${GREEN}✅ MPS (Metal Performance Shaders) available${NC}"
            echo "   Platform: Apple Silicon ($(uname -m))"
        else
            echo -e "${YELLOW}⚠️  MPS not available (Intel Mac)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  MPS requires macOS 12.0+ (current: $MACOS_VERSION)${NC}"
    fi
fi

echo -e "${GREEN}✅ CPU backend available${NC}"
echo ""

# Check library path configuration
echo "Checking library path configuration..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [[ ":$DYLD_LIBRARY_PATH:" == *":$LIBTORCH/lib:"* ]]; then
        echo -e "${GREEN}✅ DYLD_LIBRARY_PATH includes libtorch${NC}"
    else
        echo -e "${YELLOW}⚠️  DYLD_LIBRARY_PATH does not include libtorch${NC}"
        echo "   Add to your shell configuration:"
        echo "   export DYLD_LIBRARY_PATH=\$LIBTORCH/lib:\$DYLD_LIBRARY_PATH"
    fi
else
    # Linux
    if [[ ":$LD_LIBRARY_PATH:" == *":$LIBTORCH/lib:"* ]]; then
        echo -e "${GREEN}✅ LD_LIBRARY_PATH includes libtorch${NC}"
    else
        echo -e "${YELLOW}⚠️  LD_LIBRARY_PATH does not include libtorch${NC}"
        echo "   Add to your shell configuration:"
        echo "   export LD_LIBRARY_PATH=\$LIBTORCH/lib:\$LD_LIBRARY_PATH"
    fi
fi

echo ""

# Try to detect libtorch version
echo "Detecting libtorch version..."
if [ -f "$LIBTORCH/build-version" ]; then
    VERSION=$(cat "$LIBTORCH/build-version")
    echo -e "${GREEN}✅ libtorch version: $VERSION${NC}"
elif [ -f "$LIBTORCH/version.txt" ]; then
    VERSION=$(cat "$LIBTORCH/version.txt")
    echo -e "${GREEN}✅ libtorch version: $VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Unable to detect libtorch version${NC}"
fi

echo ""

# Test compilation
echo "Testing tch-rs compilation..."
echo ""

cd "$(dirname "$0")/.." # Go to rust directory

if cargo build --features tch-backend --quiet 2>&1 | grep -q "error"; then
    echo -e "${RED}❌ Compilation test failed${NC}"
    echo ""
    echo "Run the following for detailed error information:"
    echo "  cargo build --features tch-backend"
    exit 1
else
    echo -e "${GREEN}✅ tch-rs compilation successful${NC}"
fi

echo ""
echo "================================================"
echo "  Verification Summary"
echo "================================================"
echo ""
echo -e "LIBTORCH path:      ${GREEN}$LIBTORCH${NC}"
echo -e "CUDA available:     $([ "$CUDA_AVAILABLE" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${YELLOW}No${NC}")"
echo -e "MPS available:      $([ "$MPS_AVAILABLE" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${YELLOW}No${NC}")"
echo -e "CPU available:      ${GREEN}Yes${NC}"
echo -e "Compilation:        ${GREEN}Success${NC}"
echo ""
echo -e "${GREEN}✅ libtorch is properly configured!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run sanity check: cargo test --features tch-backend --test tch_sanity_check"
echo "  2. Review implementation plan: tch-rs-implementation-plan.md"
echo "  3. Proceed to Phase 2: Model Loading"
echo ""
