//! Multi-Agent Debate Module
//!
//! Implements persona-based debates for critical decisions where multiple
//! perspectives (e.g., Security Auditor vs Performance Optimizer) synthesize
//! the best approach through dialectical reasoning.

use serde::{Deserialize, Serialize};

/// A persona that Ralph can assume during debates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub name: String,
    pub role: String,
    pub priorities: Vec<String>,
    pub system_prompt: String,
}

impl Persona {
    /// Security Auditor persona - prioritizes safety and vulnerability prevention
    pub fn security_auditor() -> Self {
        Self {
            name: "Security Auditor".to_string(),
            role: "security".to_string(),
            priorities: vec![
                "Input validation".to_string(),
                "Authentication/Authorization".to_string(),
                "Data sanitization".to_string(),
                "Secure defaults".to_string(),
                "Audit logging".to_string(),
            ],
            system_prompt: r#"You are a Security Auditor reviewing code changes.
Your primary concerns are:
1. Potential vulnerabilities (injection, XSS, CSRF, etc.)
2. Authentication and authorization weaknesses
3. Data exposure risks
4. Cryptographic misuse
5. Input validation gaps

Critique the proposed changes from a security perspective. Be thorough but constructive.
Propose specific mitigations for any issues found."#
                .to_string(),
        }
    }

    /// Performance Optimizer persona - prioritizes speed and efficiency
    pub fn performance_optimizer() -> Self {
        Self {
            name: "Performance Optimizer".to_string(),
            role: "performance".to_string(),
            priorities: vec![
                "Time complexity".to_string(),
                "Memory efficiency".to_string(),
                "Cache utilization".to_string(),
                "Async optimization".to_string(),
                "Resource pooling".to_string(),
            ],
            system_prompt: r#"You are a Performance Optimizer reviewing code changes.
Your primary concerns are:
1. Algorithm efficiency (time/space complexity)
2. Memory allocation patterns
3. I/O bottlenecks and async opportunities
4. Cache-friendly data structures
5. Resource lifecycle management

Critique the proposed changes from a performance perspective. Suggest optimizations
while maintaining code clarity."#
                .to_string(),
        }
    }

    /// Architecture Purist persona - prioritizes clean design
    #[allow(dead_code)]
    pub fn architecture_purist() -> Self {
        Self {
            name: "Architecture Purist".to_string(),
            role: "architecture".to_string(),
            priorities: vec![
                "SOLID principles".to_string(),
                "Separation of concerns".to_string(),
                "Dependency inversion".to_string(),
                "Interface segregation".to_string(),
                "Testability".to_string(),
            ],
            system_prompt: r#"You are an Architecture Purist reviewing code changes.
Your primary concerns are:
1. SOLID principles adherence
2. Clean separation of concerns
3. Proper abstraction levels
4. Dependency management
5. Long-term maintainability

Critique the proposed changes from an architectural perspective. Suggest patterns
that improve the design without over-engineering."#
                .to_string(),
        }
    }

    /// User Experience Advocate persona - prioritizes usability
    #[allow(dead_code)]
    pub fn ux_advocate() -> Self {
        Self {
            name: "UX Advocate".to_string(),
            role: "ux".to_string(),
            priorities: vec![
                "Error messages clarity".to_string(),
                "API ergonomics".to_string(),
                "Documentation".to_string(),
                "Intuitive defaults".to_string(),
                "Graceful degradation".to_string(),
            ],
            system_prompt: r#"You are a UX Advocate reviewing code changes.
Your primary concerns are:
1. Clear, actionable error messages
2. Intuitive API design
3. Comprehensive documentation
4. Sensible default behaviors
5. Graceful handling of edge cases

Critique the proposed changes from a user experience perspective. Consider both
developers using this code and end-users affected by it."#
                .to_string(),
        }
    }
}

/// Result of a single debate round with critique and actionable suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateRound {
    /// The persona name (e.g., "Security Auditor")
    pub persona: String,
    /// Detailed text critique of the proposed change
    pub critique: String,
    /// Specific actionable suggestions to address identified issues
    pub suggestions: Vec<String>,
    /// Severity level of the findings
    pub severity: DebateSeverity,
}

/// Severity classification for debate findings
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DebateSeverity {
    /// Must address before proceeding (Blocker)
    Critical,
    /// Should address, but not strictly blocking (High)
    Important,
    /// Nice to have improvements (Low)
    Minor,
    /// No issues found (Pass)
    Neutral,
}

/// Final synthesis of multiple debate perspectives into a unified recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateSynthesis {
    /// Executive summary of the debate outcome
    pub summary: String,
    /// Points where multiple personas reached agreement
    pub consensus_points: Vec<String>,
    /// Situations where personas disagree on the best approach
    pub conflicts: Vec<DebateConflict>,
    /// The final authoritative recommendation (APPROVED/HOLD/PROCEED)
    pub final_recommendation: String,
    /// Prioritized action items extracted from suggestions
    pub action_items: Vec<ActionItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateConflict {
    pub topic: String,
    pub positions: Vec<(String, String)>, // (persona, position)
    pub resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub priority: u8,
    pub description: String,
    pub assigned_to: Option<String>,
}

/// The Debate orchestrator
pub struct Debate {
    personas: Vec<Persona>,
}

impl Debate {
    pub fn new() -> Self {
        Self { personas: vec![] }
    }

    /// Add a persona to the debate
    pub fn with_persona(mut self, persona: Persona) -> Self {
        self.personas.push(persona);
        self
    }

    /// Standard security vs performance debate for critical decisions
    pub fn security_vs_performance() -> Self {
        Self::new()
            .with_persona(Persona::security_auditor())
            .with_persona(Persona::performance_optimizer())
    }

    /// Full architecture review with all perspectives
    #[allow(dead_code)]
    pub fn full_review() -> Self {
        Self::new()
            .with_persona(Persona::security_auditor())
            .with_persona(Persona::performance_optimizer())
            .with_persona(Persona::architecture_purist())
            .with_persona(Persona::ux_advocate())
    }

    /// Generate debate prompts for each persona
    pub fn generate_prompts(&self, context: &str, proposed_change: &str) -> Vec<(String, String)> {
        self.personas
            .iter()
            .map(|p| {
                let prompt = format!(
                    "{}\n\n## Context\n{}\n\n## Proposed Change\n{}\n\n## Your Review\nProvide your critique as {}. Format your response as:\n\nCRITIQUE:\n[Your detailed analysis]\n\nSUGGESTIONS:\n- [Specific actionable suggestion 1]\n- [Specific actionable suggestion 2]\n\nSEVERITY: [CRITICAL|IMPORTANT|MINOR|NEUTRAL]",
                    p.system_prompt, context, proposed_change, p.name
                );
                (p.name.clone(), prompt)
            })
            .collect()
    }

    /// Parse a debate response into structured format
    pub fn parse_response(persona: &str, response: &str) -> DebateRound {
        let mut critique = String::new();
        let mut suggestions = Vec::new();
        let mut severity = DebateSeverity::Neutral;

        let mut current_section = "";
        for line in response.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("CRITIQUE:") {
                current_section = "critique";
            } else if trimmed.starts_with("SUGGESTIONS:") {
                current_section = "suggestions";
            } else if trimmed.starts_with("SEVERITY:") {
                let sev = trimmed.replace("SEVERITY:", "").trim().to_uppercase();
                severity = match sev.as_str() {
                    "CRITICAL" => DebateSeverity::Critical,
                    "IMPORTANT" => DebateSeverity::Important,
                    "MINOR" => DebateSeverity::Minor,
                    _ => DebateSeverity::Neutral,
                };
            } else {
                match current_section {
                    "critique" => {
                        critique.push_str(trimmed);
                        critique.push('\n');
                    }
                    "suggestions" => {
                        if trimmed.starts_with('-') || trimmed.starts_with('*') {
                            suggestions.push(trimmed[1..].trim().to_string());
                        }
                    }
                    _ => {}
                }
            }
        }

        DebateRound {
            persona: persona.to_string(),
            critique: critique.trim().to_string(),
            suggestions,
            severity,
        }
    }

    /// Synthesize multiple debate rounds into a final recommendation
    pub fn synthesize(rounds: &[DebateRound]) -> DebateSynthesis {
        let consensus_points = Vec::new();
        let conflicts = Vec::new();
        let mut action_items = Vec::new();

        // Collect all suggestions
        let _all_suggestions: Vec<_> = rounds
            .iter()
            .flat_map(|r| r.suggestions.iter().map(|s| (r.persona.clone(), s.clone())))
            .collect();

        // Find the most severe issues
        let has_critical = rounds
            .iter()
            .any(|r| r.severity == DebateSeverity::Critical);
        let has_important = rounds
            .iter()
            .any(|r| r.severity == DebateSeverity::Important);

        // Generate action items from suggestions with priority based on severity
        for round in rounds {
            let priority = match round.severity {
                DebateSeverity::Critical => 1,
                DebateSeverity::Important => 2,
                DebateSeverity::Minor => 3,
                DebateSeverity::Neutral => 4,
            };

            for suggestion in &round.suggestions {
                action_items.push(ActionItem {
                    priority,
                    description: suggestion.clone(),
                    assigned_to: Some(round.persona.clone()),
                });
            }
        }

        // Sort action items by priority
        action_items.sort_by_key(|a| a.priority);

        let summary = if has_critical {
            "Critical issues identified that must be addressed before proceeding.".to_string()
        } else if has_important {
            "Important concerns raised that should be addressed.".to_string()
        } else {
            "No significant issues identified. Proceed with minor adjustments if suggested."
                .to_string()
        };

        let final_recommendation = if has_critical {
            "HOLD: Address critical issues before implementation.".to_string()
        } else if has_important {
            "PROCEED WITH CAUTION: Implement suggested improvements.".to_string()
        } else {
            "APPROVED: Proceed with implementation.".to_string()
        };

        DebateSynthesis {
            summary,
            consensus_points,
            conflicts,
            final_recommendation,
            action_items,
        }
    }
}
