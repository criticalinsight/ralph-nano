//! Automated Benchmarking Module
//!
//! Runs benchmarks after performance-related changes, stores results
//! in local files, and tracks regressions over time.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// A single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub duration_ns: u64,
    pub iterations: u64,
    pub throughput: Option<f64>,  // ops/sec or bytes/sec
    pub memory_bytes: Option<u64>,
    pub timestamp: String,
    pub commit_hash: String,
    pub tags: Vec<String>,
}

/// Aggregated benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    pub name: String,
    pub mean_ns: f64,
    pub std_dev_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub samples: usize,
}

/// Comparison between two benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub name: String,
    pub baseline_ns: f64,
    pub current_ns: f64,
    pub change_percent: f64,
    pub regression: bool,
    pub significant: bool,  // > 5% change
}

/// Benchmark suite definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub benchmarks: Vec<String>,
    pub command: String,
    pub parser: BenchmarkParser,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BenchmarkParser {
    Criterion,    // Rust criterion output
    Hyperfine,    // Generic command timing
    Custom,       // Custom format
    Jest,         // JavaScript jest benchmarks
    Pytest,       // Python pytest-benchmark
}

/// The benchmark runner and tracker
pub struct Benchmarker {
    workspace: String,
    results_dir: String,
    history: Vec<BenchmarkResult>,
}

impl Benchmarker {
    pub fn new(workspace: &str) -> Self {
        let results_dir = format!("{}/.ralph/benchmarks", workspace);
        std::fs::create_dir_all(&results_dir).ok();
        
        Self {
            workspace: workspace.to_string(),
            results_dir,
            history: Vec::new(),
        }
    }

    /// Detect available benchmark suites based on project type
    pub fn detect_suites(&self) -> Vec<BenchmarkSuite> {
        let mut suites = Vec::new();

        // Rust: Check for benches directory or criterion
        if Path::new(&format!("{}/benches", self.workspace)).exists() {
            suites.push(BenchmarkSuite {
                name: "Rust Benchmarks".to_string(),
                benchmarks: vec![],
                command: "cargo bench".to_string(),
                parser: BenchmarkParser::Criterion,
            });
        }

        // Check for hyperfine script
        if Path::new(&format!("{}/benchmark.sh", self.workspace)).exists() {
            suites.push(BenchmarkSuite {
                name: "Custom Benchmarks".to_string(),
                benchmarks: vec![],
                command: "./benchmark.sh".to_string(),
                parser: BenchmarkParser::Hyperfine,
            });
        }

        // JavaScript: Check for benchmark scripts
        if Path::new(&format!("{}/package.json", self.workspace)).exists() {
            if let Ok(content) = std::fs::read_to_string(format!("{}/package.json", self.workspace)) {
                if content.contains("\"bench\"") || content.contains("\"benchmark\"") {
                    suites.push(BenchmarkSuite {
                        name: "Node Benchmarks".to_string(),
                        benchmarks: vec![],
                        command: "npm run bench".to_string(),
                        parser: BenchmarkParser::Jest,
                    });
                }
            }
        }

        // Python: Check for pytest-benchmark
        if Path::new(&format!("{}/pytest.ini", self.workspace)).exists() ||
           Path::new(&format!("{}/pyproject.toml", self.workspace)).exists() {
            suites.push(BenchmarkSuite {
                name: "Python Benchmarks".to_string(),
                benchmarks: vec![],
                command: "pytest --benchmark-only".to_string(),
                parser: BenchmarkParser::Pytest,
            });
        }

        suites
    }

    /// Run a benchmark suite
    pub fn run_suite(&mut self, suite: &BenchmarkSuite) -> Result<Vec<BenchmarkResult>> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(&suite.command)
            .current_dir(&self.workspace)
            .output()
            .context("Failed to run benchmark command")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let results = self.parse_output(&stdout, suite.parser)?;
        
        // Add to history
        self.history.extend(results.clone());
        
        Ok(results)
    }

    /// Parse benchmark output based on parser type
    fn parse_output(&self, output: &str, parser: BenchmarkParser) -> Result<Vec<BenchmarkResult>> {
        let commit_hash = self.get_current_commit()?;
        let timestamp = Utc::now().to_rfc3339();

        match parser {
            BenchmarkParser::Criterion => self.parse_criterion(output, &commit_hash, &timestamp),
            BenchmarkParser::Hyperfine => self.parse_hyperfine(output, &commit_hash, &timestamp),
            _ => Ok(vec![]),
        }
    }

    fn parse_criterion(&self, output: &str, commit: &str, timestamp: &str) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        // Parse criterion output format
        // Example: "test benchmark_name ... bench:     123,456 ns/iter (+/- 1,234)"
        let re = regex::Regex::new(r"(?m)^(.+?)\s+time:\s+\[([0-9.]+)\s+(\w+)")?;
        
        for cap in re.captures_iter(output) {
            let name = cap.get(1).map(|m| m.as_str().trim()).unwrap_or("");
            let time_str = cap.get(2).map(|m| m.as_str()).unwrap_or("0");
            let unit = cap.get(3).map(|m| m.as_str()).unwrap_or("ns");
            
            let time: f64 = time_str.replace(',', "").parse().unwrap_or(0.0);
            let duration_ns = match unit {
                "ps" => (time / 1000.0) as u64,
                "ns" => time as u64,
                "µs" | "us" => (time * 1_000.0) as u64,
                "ms" => (time * 1_000_000.0) as u64,
                "s" => (time * 1_000_000_000.0) as u64,
                _ => time as u64,
            };

            results.push(BenchmarkResult {
                name: name.to_string(),
                duration_ns,
                iterations: 1,
                throughput: None,
                memory_bytes: None,
                timestamp: timestamp.to_string(),
                commit_hash: commit.to_string(),
                tags: vec!["criterion".to_string()],
            });
        }

        Ok(results)
    }

    fn parse_hyperfine(&self, output: &str, commit: &str, timestamp: &str) -> Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        // Parse hyperfine JSON output or summary
        // Example: "Benchmark 1: command  Time (mean ± σ):     123.4 ms ±   5.6 ms"
        let re = regex::Regex::new(r"Time \(mean.*?\):\s+([0-9.]+)\s+(\w+)")?;
        
        for cap in re.captures_iter(output) {
            let time_str = cap.get(1).map(|m| m.as_str()).unwrap_or("0");
            let unit = cap.get(2).map(|m| m.as_str()).unwrap_or("ms");
            
            let time: f64 = time_str.parse().unwrap_or(0.0);
            let duration_ns = match unit {
                "ns" => time as u64,
                "µs" | "us" => (time * 1_000.0) as u64,
                "ms" => (time * 1_000_000.0) as u64,
                "s" => (time * 1_000_000_000.0) as u64,
                _ => (time * 1_000_000.0) as u64,
            };

            results.push(BenchmarkResult {
                name: "hyperfine".to_string(),
                duration_ns,
                iterations: 1,
                throughput: None,
                memory_bytes: None,
                timestamp: timestamp.to_string(),
                commit_hash: commit.to_string(),
                tags: vec!["hyperfine".to_string()],
            });
        }

        Ok(results)
    }

    fn get_current_commit(&self) -> Result<String> {
        let output = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(&self.workspace)
            .output()?;
        
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Compare current results with baseline
    pub fn compare_with_baseline(&self, baseline_commit: &str) -> Vec<BenchmarkComparison> {
        let baseline: HashMap<String, f64> = self.history
            .iter()
            .filter(|r| r.commit_hash == baseline_commit)
            .map(|r| (r.name.clone(), r.duration_ns as f64))
            .collect();

        let current_commit = self.get_current_commit().unwrap_or_default();
        let current: HashMap<String, f64> = self.history
            .iter()
            .filter(|r| r.commit_hash == current_commit)
            .map(|r| (r.name.clone(), r.duration_ns as f64))
            .collect();

        let mut comparisons = Vec::new();
        for (name, baseline_ns) in &baseline {
            if let Some(&current_ns) = current.get(name) {
                let change_percent = ((current_ns - baseline_ns) / baseline_ns) * 100.0;
                comparisons.push(BenchmarkComparison {
                    name: name.clone(),
                    baseline_ns: *baseline_ns,
                    current_ns,
                    change_percent,
                    regression: change_percent > 0.0,
                    significant: change_percent.abs() > 5.0,
                });
            }
        }

        comparisons
    }

    /// Generate a benchmark report
    pub fn generate_report(&self) -> String {
        let mut report = String::from("# Benchmark Report\n\n");
        
        // Group by commit
        let mut by_commit: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.history {
            by_commit.entry(result.commit_hash.clone()).or_default().push(result);
        }

        for (commit, results) in by_commit.iter().take(5) {
            report.push_str(&format!("## Commit: {}\n\n", commit));
            report.push_str("| Benchmark | Duration | Throughput |\n");
            report.push_str("|-----------|----------|------------|\n");
            
            for r in results {
                let duration = if r.duration_ns > 1_000_000_000 {
                    format!("{:.2}s", r.duration_ns as f64 / 1_000_000_000.0)
                } else if r.duration_ns > 1_000_000 {
                    format!("{:.2}ms", r.duration_ns as f64 / 1_000_000.0)
                } else if r.duration_ns > 1_000 {
                    format!("{:.2}µs", r.duration_ns as f64 / 1_000.0)
                } else {
                    format!("{}ns", r.duration_ns)
                };
                
                let throughput = r.throughput
                    .map(|t| format!("{:.2} ops/s", t))
                    .unwrap_or_else(|| "-".to_string());
                    
                report.push_str(&format!("| {} | {} | {} |\n", r.name, duration, throughput));
            }
            report.push('\n');
        }

        report
    }

    /// Save results to disk
    pub fn save_results(&self) -> Result<()> {
        let path = format!("{}/history.json", self.results_dir);
        let json = serde_json::to_string_pretty(&self.history)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load results from disk
    pub fn load_results(&mut self) -> Result<()> {
        let path = format!("{}/history.json", self.results_dir);
        if Path::new(&path).exists() {
            let content = std::fs::read_to_string(path)?;
            self.history = serde_json::from_str(&content)?;
        }
        Ok(())
    }
}

/// Quick timing utility for inline benchmarks
pub fn time_block<F, T>(name: &str, f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("⏱️  {}: {:?}", name, duration);
    (result, duration)
}
