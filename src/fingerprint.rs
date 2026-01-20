// src/fingerprint.rs - Project Fingerprinting and Dependency Awareness for Ralph-Nano v1.2.0

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFingerprint {
    pub project_type: ProjectType,
    pub tech_stack: Vec<String>,
    pub dependencies: Vec<Dependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ProjectType {
    RustCli,
    RustLib,
    RustWeb,
    NodeJs,
    TypeScript,
    Python,
    PythonMl,
    NextJs,
    React,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: Option<String>,
}

impl ProjectFingerprint {
    pub fn detect(root: &Path) -> Self {
        let mut fingerprint = Self {
            project_type: ProjectType::Unknown,
            tech_stack: Vec::new(),
            dependencies: Vec::new(),
        };

        // Check for Rust project
        let cargo_toml = root.join("Cargo.toml");
        if cargo_toml.exists() {
            fingerprint.tech_stack.push("Rust".to_string());
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                fingerprint.project_type = Self::detect_rust_type(&content);
                fingerprint.dependencies = Self::parse_cargo_deps(&content);
            }
        }

        // Check for Node.js project
        let package_json = root.join("package.json");
        if package_json.exists() {
            fingerprint.tech_stack.push("Node.js".to_string());
            if let Ok(content) = fs::read_to_string(&package_json) {
                if let Ok(pkg) = serde_json::from_str::<serde_json::Value>(&content) {
                    fingerprint.project_type = Self::detect_node_type(&pkg);
                    fingerprint.dependencies = Self::parse_npm_deps(&pkg);
                }
            }
        }

        // Check for Python project
        let requirements = root.join("requirements.txt");
        let pyproject = root.join("pyproject.toml");
        if requirements.exists() || pyproject.exists() {
            fingerprint.tech_stack.push("Python".to_string());
            if requirements.exists() {
                if let Ok(content) = fs::read_to_string(&requirements) {
                    fingerprint.dependencies = Self::parse_requirements(&content);
                }
            }
            fingerprint.project_type = Self::detect_python_type(&fingerprint.dependencies);
        }

        fingerprint
    }

    fn detect_rust_type(cargo_content: &str) -> ProjectType {
        if cargo_content.contains("actix")
            || cargo_content.contains("axum")
            || cargo_content.contains("warp")
        {
            ProjectType::RustWeb
        } else if cargo_content.contains("[lib]") {
            ProjectType::RustLib
        } else {
            ProjectType::RustCli
        }
    }

    fn detect_node_type(pkg: &serde_json::Value) -> ProjectType {
        let deps = pkg.get("dependencies").and_then(|d| d.as_object());
        if let Some(deps) = deps {
            if deps.contains_key("next") {
                return ProjectType::NextJs;
            }
            if deps.contains_key("react") {
                return ProjectType::React;
            }
        }
        if pkg
            .get("devDependencies")
            .and_then(|d| d.get("typescript"))
            .is_some()
        {
            return ProjectType::TypeScript;
        }
        ProjectType::NodeJs
    }

    fn detect_python_type(deps: &[Dependency]) -> ProjectType {
        let ml_libs = ["torch", "tensorflow", "sklearn", "numpy", "pandas", "keras"];
        if deps.iter().any(|d| ml_libs.contains(&d.name.as_str())) {
            ProjectType::PythonMl
        } else {
            ProjectType::Python
        }
    }

    fn parse_cargo_deps(content: &str) -> Vec<Dependency> {
        let mut deps = Vec::new();
        let mut in_deps = false;
        for line in content.lines() {
            if line.starts_with("[dependencies]") || line.starts_with("[dev-dependencies]") {
                in_deps = true;
                continue;
            }
            if line.starts_with('[') {
                in_deps = false;
            }
            if in_deps && line.contains('=') {
                let parts: Vec<&str> = line.splitn(2, '=').collect();
                if parts.len() == 2 {
                    let name = parts[0].trim().to_string();
                    let version = parts[1].trim().trim_matches('"').to_string();
                    deps.push(Dependency {
                        name,
                        version: Some(version),
                    });
                }
            }
        }
        deps
    }

    fn parse_npm_deps(pkg: &serde_json::Value) -> Vec<Dependency> {
        let mut deps = Vec::new();
        for key in ["dependencies", "devDependencies"] {
            if let Some(d) = pkg.get(key).and_then(|v| v.as_object()) {
                for (name, version) in d {
                    deps.push(Dependency {
                        name: name.clone(),
                        version: version.as_str().map(|s| s.to_string()),
                    });
                }
            }
        }
        deps
    }

    fn parse_requirements(content: &str) -> Vec<Dependency> {
        content
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .map(|l| {
                let parts: Vec<&str> = l.splitn(2, ['=', '>', '<']).collect();
                Dependency {
                    name: parts[0].trim().to_string(),
                    version: parts
                        .get(1)
                        .map(|s| s.trim_matches(|c| c == '=' || c == ' ').to_string()),
                }
            })
            .collect()
    }

    pub fn specialized_prompt(&self) -> String {
        match self.project_type {
            ProjectType::RustCli => "This is a Rust CLI application. Use idiomatic Rust patterns, prefer clap for arg parsing, and ensure proper error handling with anyhow or thiserror.".to_string(),
            ProjectType::RustLib => "This is a Rust library. Focus on clean public APIs, comprehensive documentation, and avoid unnecessary dependencies.".to_string(),
            ProjectType::RustWeb => "This is a Rust web application. Use async patterns, implement proper middleware, and follow RESTful conventions.".to_string(),
            ProjectType::NextJs => "This is a Next.js application. Use App Router patterns, Server Components where appropriate, and follow React best practices.".to_string(),
            ProjectType::React => "This is a React application. Use functional components with hooks, implement proper state management, and ensure accessibility.".to_string(),
            ProjectType::TypeScript => "This is a TypeScript project. Ensure strict type safety, use interfaces for data shapes, and avoid 'any' types.".to_string(),
            ProjectType::NodeJs => "This is a Node.js project. Use modern ES modules, handle async operations properly, and implement robust error handling.".to_string(),
            ProjectType::Python => "This is a Python project. Follow PEP 8, use type hints, and prefer dataclasses or Pydantic for data modeling.".to_string(),
            ProjectType::PythonMl => "This is a Python ML project. Use numpy/pandas efficiently, implement proper data validation, and document model assumptions.".to_string(),
            ProjectType::Unknown => "No specific project type detected. Apply general software engineering best practices.".to_string(),
        }
    }

    pub fn dependency_context(&self) -> String {
        if self.dependencies.is_empty() {
            return String::new();
        }

        let top_deps: Vec<String> = self
            .dependencies
            .iter()
            .take(10)
            .map(|d| d.name.clone())
            .collect();

        format!("KEY DEPENDENCIES: {}", top_deps.join(", "))
    }
}
