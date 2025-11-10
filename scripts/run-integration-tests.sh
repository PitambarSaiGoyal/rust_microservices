#!/usr/bin/env bash
#
# Comprehensive Integration Test Runner for LlamaFirewall
#
# This script orchestrates the full integration test suite including:
# - Unit tests
# - Integration tests
# - Stress tests
# - Accuracy validation
# - Performance benchmarks
#
# Usage:
#   ./rust/scripts/run-integration-tests.sh [options]
#
# Options:
#   --quick       Run only fast tests (skip stress tests)
#   --full        Run all tests including long-running stress tests
#   --onnx-only   Test only ONNX backend
#   --tch-only    Test only tch-rs backend
#   --accuracy    Run accuracy validation
#   --stress      Run stress tests only

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
QUICK_MODE=false
FULL_MODE=false
ONNX_ONLY=false
TCH_ONLY=false
ACCURACY_MODE=false
STRESS_MODE=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        --onnx-only)
            ONNX_ONLY=true
            shift
            ;;
        --tch-only)
            TCH_ONLY=true
            shift
            ;;
        --accuracy)
            ACCURACY_MODE=true
            shift
            ;;
        --stress)
            STRESS_MODE=true
            shift
            ;;
        *)
            # unknown option
            ;;
    esac
done

# Default to quick mode if nothing specified
if [ "$FULL_MODE" = false ] && [ "$ACCURACY_MODE" = false ] && [ "$STRESS_MODE" = false ]; then
    QUICK_MODE=true
fi

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Determine which backends to test
ONNX_FEATURES=""
TCH_FEATURES=""

if [ "$TCH_ONLY" = false ]; then
    ONNX_FEATURES="--features onnx-backend"
fi

if [ "$ONNX_ONLY" = false ]; then
    TCH_FEATURES="--features tch-backend"
fi

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    echo -e "\n${YELLOW}Running: $test_name${NC}"

    if eval "$test_command"; then
        print_success "$test_name passed"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "$test_name failed"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    SKIPPED_TESTS=$((SKIPPED_TESTS + 1))

    print_warning "Skipped: $test_name ($reason)"
}

# Main test execution
main() {
    print_header "LlamaFirewall Integration Test Suite"

    echo "Configuration:"
    echo "  Quick Mode:    $QUICK_MODE"
    echo "  Full Mode:     $FULL_MODE"
    echo "  ONNX Only:     $ONNX_ONLY"
    echo "  tch-rs Only:   $TCH_ONLY"
    echo "  Accuracy:      $ACCURACY_MODE"
    echo "  Stress:        $STRESS_MODE"

    # Phase 1: Build tests
    if [ "$STRESS_MODE" = false ]; then
        print_header "Phase 1: Build Verification"

        if [ "$TCH_ONLY" = false ]; then
            run_test "Build ONNX backend" \
                "cargo build --workspace --features onnx-backend"
        fi

        if [ "$ONNX_ONLY" = false ]; then
            if [ -n "${LIBTORCH:-}" ]; then
                run_test "Build tch-rs backend" \
                    "cargo build --workspace --features tch-backend"
            else
                skip_test "Build tch-rs backend" "LIBTORCH not set"
            fi
        fi
    fi

    # Phase 2: Unit tests
    if [ "$STRESS_MODE" = false ]; then
        print_header "Phase 2: Unit Tests"

        if [ "$TCH_ONLY" = false ]; then
            run_test "ONNX unit tests" \
                "cargo test --lib $ONNX_FEATURES"
        fi

        if [ "$ONNX_ONLY" = false ]; then
            if [ -n "${LIBTORCH:-}" ]; then
                run_test "tch-rs unit tests" \
                    "cargo test --lib $TCH_FEATURES"
            else
                skip_test "tch-rs unit tests" "LIBTORCH not set"
            fi
        fi
    fi

    # Phase 3: Integration tests
    if [ "$STRESS_MODE" = false ] && ([ "$QUICK_MODE" = true ] || [ "$FULL_MODE" = true ]); then
        print_header "Phase 3: Integration Tests"

        if [ "$TCH_ONLY" = false ]; then
            run_test "ONNX integration tests" \
                "cargo test --test integration_suite $ONNX_FEATURES"
        fi

        if [ "$ONNX_ONLY" = false ]; then
            if [ -n "${LIBTORCH:-}" ]; then
                run_test "tch-rs integration tests" \
                    "cargo test --test integration_suite $TCH_FEATURES"
            else
                skip_test "tch-rs integration tests" "LIBTORCH not set"
            fi
        fi

        # Cross-backend tests
        if [ "$ONNX_ONLY" = false ] && [ "$TCH_ONLY" = false ] && [ -n "${LIBTORCH:-}" ]; then
            run_test "Cross-backend consistency tests" \
                "cargo test --test integration_suite --features onnx-backend,tch-backend"
        fi
    fi

    # Phase 4: Stress tests
    if [ "$FULL_MODE" = true ] || [ "$STRESS_MODE" = true ]; then
        print_header "Phase 4: Stress Tests"

        if [ "$TCH_ONLY" = false ]; then
            run_test "ONNX memory leak detection" \
                "cargo test --test stress_tests $ONNX_FEATURES -- --ignored --nocapture test_onnx_memory_leak"

            run_test "ONNX sustained load test" \
                "cargo test --test stress_tests $ONNX_FEATURES -- --ignored --nocapture test_onnx_sustained_load"

            run_test "ONNX error recovery test" \
                "cargo test --test stress_tests $ONNX_FEATURES -- --ignored --nocapture test_onnx_error_recovery"
        fi

        if [ "$ONNX_ONLY" = false ] && [ -n "${LIBTORCH:-}" ]; then
            run_test "tch-rs memory leak detection" \
                "cargo test --test stress_tests $TCH_FEATURES -- --ignored --nocapture test_tch_memory_leak"

            run_test "tch-rs sustained load test" \
                "cargo test --test stress_tests $TCH_FEATURES -- --ignored --nocapture test_tch_sustained_load"

            run_test "tch-rs error recovery test" \
                "cargo test --test stress_tests $TCH_FEATURES -- --ignored --nocapture test_tch_error_recovery"
        fi
    fi

    # Phase 5: Accuracy validation
    if [ "$ACCURACY_MODE" = true ]; then
        print_header "Phase 5: Accuracy Validation"

        if [ "$ONNX_ONLY" = false ] && [ "$TCH_ONLY" = false ]; then
            if command -v python3 &> /dev/null; then
                run_test "Backend accuracy comparison (1000 samples)" \
                    "python3 scripts/validate_accuracy.py --samples 1000"
            else
                skip_test "Accuracy validation" "Python 3 not found"
            fi
        else
            skip_test "Accuracy validation" "Requires both backends"
        fi
    fi

    # Phase 6: Performance benchmarks
    if [ "$FULL_MODE" = true ]; then
        print_header "Phase 6: Performance Benchmarks"

        if [ "$TCH_ONLY" = false ]; then
            # Run individual ONNX benchmarks (avoid comprehensive_bench due to mutex issue)
            run_test "ONNX performance benchmarks" \
                "cargo bench --bench promptguard_bench --features onnx-backend -- --quick"
        fi

        if [ "$ONNX_ONLY" = false ] && [ -n "${LIBTORCH:-}" ]; then
            run_test "tch-rs performance benchmarks" \
                "cargo bench --bench tch_bench --features tch-backend -- --quick"
        fi
    fi

    # Summary
    print_header "Test Summary"

    echo "Total Tests:   $TOTAL_TESTS"
    echo -e "${GREEN}Passed:        $PASSED_TESTS${NC}"

    if [ $FAILED_TESTS -gt 0 ]; then
        echo -e "${RED}Failed:        $FAILED_TESTS${NC}"
    else
        echo "Failed:        $FAILED_TESTS"
    fi

    if [ $SKIPPED_TESTS -gt 0 ]; then
        echo -e "${YELLOW}Skipped:       $SKIPPED_TESTS${NC}"
    else
        echo "Skipped:       $SKIPPED_TESTS"
    fi

    # Exit code
    if [ $FAILED_TESTS -gt 0 ]; then
        print_error "Some tests failed"
        exit 1
    elif [ $PASSED_TESTS -eq 0 ]; then
        print_warning "No tests were run"
        exit 2
    else
        print_success "All tests passed!"
        exit 0
    fi
}

# Run main function
main
