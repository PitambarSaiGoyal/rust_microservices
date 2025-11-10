#!/usr/bin/env python3
"""
Compare Python and Rust benchmark results.

This script loads benchmark results from both implementations and generates
a comparison report showing speedup ratios and performance differences.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import sys


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON from {filepath}: {e}")
        sys.exit(1)


def parse_criterion_results(criterion_dir: Path) -> Dict[str, float]:
    """Parse Criterion benchmark results from Rust
    
    Args:
        criterion_dir: Path to target/criterion directory
        
    Returns:
        Dict mapping benchmark name to mean time in ms
    """
    results = {}
    
    # Look for estimate JSON files in criterion output
    for bench_group in criterion_dir.glob("*/"):
        if not bench_group.is_dir():
            continue
            
        for bench_name in bench_group.glob("*/"):
            if not bench_name.is_dir():
                continue
                
            estimates_file = bench_name / "base" / "estimates.json"
            if not estimates_file.exists():
                estimates_file = bench_name / "new" / "estimates.json"
            
            if estimates_file.exists():
                try:
                    with open(estimates_file, 'r') as f:
                        data = json.load(f)
                        # Criterion stores time in nanoseconds
                        mean_ns = data.get("mean", {}).get("point_estimate", 0)
                        mean_ms = mean_ns / 1_000_000  # Convert ns to ms
                        
                        full_name = f"{bench_group.name}/{bench_name.name}"
                        results[full_name] = mean_ms
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️  Warning: Could not parse {estimates_file}: {e}")
    
    return results


def format_speedup(python_ms: float, rust_ms: float) -> str:
    """Format speedup ratio as a string"""
    if rust_ms == 0:
        return "N/A"
    
    ratio = python_ms / rust_ms
    if ratio > 1:
        return f"{ratio:.2f}x faster ✅"
    elif ratio < 1:
        return f"{1/ratio:.2f}x slower ❌"
    else:
        return "Same speed"


def compare_benchmarks(python_results: Dict[str, Any], rust_results: Dict[str, float]):
    """Generate comparison report"""
    
    print("\n" + "=" * 80)
    print("🦙 LLAMAFIREWALL PERFORMANCE COMPARISON: PYTHON VS RUST")
    print("=" * 80)
    
    # 1. Single prompt comparison
    print("\n📊 SINGLE PROMPT PERFORMANCE")
    print("-" * 80)
    
    python_single = python_results.get("benchmarks", {}).get("single_overall", {})
    python_mean = python_single.get("mean_ms", 0)
    
    # Try to find corresponding Rust benchmark
    rust_single = None
    for key, value in rust_results.items():
        if "single_prompt" in key.lower():
            rust_single = value
            break
    
    print(f"Python:  {python_mean:.2f}ms (mean)")
    if rust_single:
        print(f"Rust:    {rust_single:.2f}ms (mean)")
        print(f"Speedup: {format_speedup(python_mean, rust_single)}")
    else:
        print("Rust:    [No matching benchmark found]")
    
    # 2. Batch comparison
    print("\n📊 BATCH PROCESSING (8 prompts)")
    print("-" * 80)
    
    python_batch = python_results.get("benchmarks", {}).get("batch", {})
    python_batch_total = python_batch.get("mean_total_ms", 0)
    python_batch_per = python_batch.get("mean_per_prompt_ms", 0)
    
    rust_batch = None
    for key, value in rust_results.items():
        if "batch" in key.lower() and "8" in key:
            rust_batch = value
            break
    
    print(f"Python:  {python_batch_total:.2f}ms total, {python_batch_per:.2f}ms per prompt")
    if rust_batch:
        rust_batch_per = rust_batch / 8
        print(f"Rust:    {rust_batch:.2f}ms total, {rust_batch_per:.2f}ms per prompt")
        print(f"Speedup: {format_speedup(python_batch_per, rust_batch_per)}")
    else:
        print("Rust:    [No matching benchmark found]")
    
    # 3. Overall summary
    print("\n📈 SUMMARY")
    print("=" * 80)
    
    if rust_single and python_mean:
        speedup = python_mean / rust_single
        speedup_percent = (speedup - 1) * 100
        
        if speedup > 1:
            print(f"✅ Rust is {speedup:.2f}x faster than Python ({speedup_percent:.1f}% improvement)")
        elif speedup < 1:
            slowdown = 1 / speedup
            slowdown_percent = (slowdown - 1) * 100
            print(f"❌ Rust is {slowdown:.2f}x slower than Python ({slowdown_percent:.1f}% regression)")
        else:
            print("⚖️  Rust and Python have similar performance")
        
        # Target analysis
        print("\n🎯 TARGET ANALYSIS")
        print("-" * 80)
        print(f"Python baseline:     {python_mean:.2f}ms")
        print(f"Rust actual:         {rust_single:.2f}ms")
        print(f"Target (50-80ms):    {'✅ MET' if 50 <= rust_single <= 80 else '❌ NOT MET'}")
        print(f"Target (match Python): {'✅ MET' if rust_single <= python_mean * 1.1 else '❌ NOT MET'}")
    else:
        print("⚠️  Insufficient data for comparison")
    
    # 4. Detailed breakdown
    print("\n📋 DETAILED BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Benchmark':<40} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<20}")
    print("-" * 80)
    
    # Single prompts
    if "single_prompts" in python_results.get("benchmarks", {}):
        for i, prompt_result in enumerate(python_results["benchmarks"]["single_prompts"], 1):
            prompt_name = f"Prompt {i}"
            python_time = prompt_result.get("mean_ms", 0)
            
            # Try to find matching Rust result
            rust_time = None
            for key, value in rust_results.items():
                if f"by_type" in key and str(i-1) in key:
                    rust_time = value
                    break
            
            if rust_time:
                speedup = format_speedup(python_time, rust_time)
                print(f"{prompt_name:<40} {python_time:<15.2f} {rust_time:<15.2f} {speedup:<20}")
            else:
                print(f"{prompt_name:<40} {python_time:<15.2f} {'N/A':<15} {'N/A':<20}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare Python and Rust benchmark results")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("benchmarks/python_results.json"),
        help="Path to Python benchmark results JSON"
    )
    parser.add_argument(
        "--rust",
        type=Path,
        default=Path("rust/target/criterion"),
        help="Path to Rust criterion results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save comparison report to file"
    )
    
    args = parser.parse_args()
    
    # Load Python results
    print(f"📂 Loading Python results from: {args.python}")
    python_results = load_json(args.python)
    
    # Load Rust results
    print(f"📂 Loading Rust results from: {args.rust}")
    if args.rust.is_dir():
        rust_results = parse_criterion_results(args.rust)
        if not rust_results:
            print(f"⚠️  Warning: No Criterion results found in {args.rust}")
            print("    Make sure to run: cargo bench --bench comprehensive_bench")
    else:
        print(f"❌ Error: Rust criterion directory not found: {args.rust}")
        sys.exit(1)
    
    # Generate comparison
    compare_benchmarks(python_results, rust_results)
    
    # Save to file if requested
    if args.output:
        # TODO: Implement saving report to file
        print(f"\n💾 Report saved to: {args.output}")


if __name__ == "__main__":
    main()
