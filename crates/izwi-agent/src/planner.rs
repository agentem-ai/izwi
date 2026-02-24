use crate::errors::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlanningMode {
    Off,
    #[default]
    Auto,
    On,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub mode: PlanningMode,
    pub steps: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum PlannerDecision {
    DirectRespond,
    PlanThenAct(PlanSummary),
}

#[derive(Debug, Clone)]
pub struct PlannerContext {
    pub user_text: String,
    pub planning_mode: PlanningMode,
    pub tools_enabled: bool,
    pub available_tool_names: Vec<String>,
}

pub trait Planner: Send + Sync {
    fn decide(&self, ctx: &PlannerContext) -> Result<PlannerDecision>;
}

#[derive(Debug, Clone, Default)]
pub struct SimplePlanner;

impl Planner for SimplePlanner {
    fn decide(&self, ctx: &PlannerContext) -> Result<PlannerDecision> {
        if ctx.planning_mode == PlanningMode::Off {
            return Ok(PlannerDecision::DirectRespond);
        }

        let lower = ctx.user_text.to_lowercase();
        let should_plan = match ctx.planning_mode {
            PlanningMode::On => true,
            PlanningMode::Off => false,
            PlanningMode::Auto => {
                lower.contains("research")
                    || lower.contains("compare")
                    || lower.contains("teach")
                    || lower.contains("learn")
                    || lower.contains("language")
                    || lower.contains("lesson")
                    || lower.contains("plan")
                    || lower.starts_with("how do i")
                    || lower.starts_with("how can i")
            }
        };

        if !should_plan {
            return Ok(PlannerDecision::DirectRespond);
        }

        let mut steps =
            if lower.contains("teach") || lower.contains("learn") || lower.contains("language") {
                vec![
                    "Identify the user's learning goal and current level.".to_string(),
                    "Teach in short steps with simple examples and checks.".to_string(),
                    "End with a small practice task or recap.".to_string(),
                ]
            } else if lower.contains("research") || lower.contains("compare") {
                vec![
                    "Clarify the topic and what the user wants compared or researched.".to_string(),
                    "Gather facts from available tools or prior context.".to_string(),
                    "Return a concise summary with clear next steps.".to_string(),
                ]
            } else {
                vec![
                    "Break the request into a few concrete steps.".to_string(),
                    "Work through the steps and keep the user updated.".to_string(),
                    "Summarize the result and next actions.".to_string(),
                ]
            };

        if ctx.tools_enabled && !ctx.available_tool_names.is_empty() {
            steps.insert(
                1,
                "Use tools only when needed, then integrate results.".to_string(),
            );
        }

        Ok(PlannerDecision::PlanThenAct(PlanSummary {
            mode: ctx.planning_mode,
            steps,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_mode_plans_for_teaching_requests() {
        let planner = SimplePlanner;
        let decision = planner
            .decide(&PlannerContext {
                user_text: "Teach me basic Spanish greetings".to_string(),
                planning_mode: PlanningMode::Auto,
                tools_enabled: true,
                available_tool_names: vec!["time".to_string()],
            })
            .expect("planner should succeed");

        match decision {
            PlannerDecision::PlanThenAct(plan) => {
                assert!(!plan.steps.is_empty());
            }
            PlannerDecision::DirectRespond => panic!("expected planning"),
        }
    }

    #[test]
    fn off_mode_disables_planning() {
        let planner = SimplePlanner;
        let decision = planner
            .decide(&PlannerContext {
                user_text: "research rust lifetimes".to_string(),
                planning_mode: PlanningMode::Off,
                tools_enabled: true,
                available_tool_names: vec!["time".to_string()],
            })
            .expect("planner should succeed");

        assert!(matches!(decision, PlannerDecision::DirectRespond));
    }
}
