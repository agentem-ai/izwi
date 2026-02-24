use crate::planner::PlanningMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub memory: bool,
    pub planning: bool,
    pub tools: bool,
    pub voice: bool,
}

impl Default for AgentCapabilities {
    fn default() -> Self {
        Self {
            memory: true,
            planning: true,
            tools: true,
            voice: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub default_model: String,
    #[serde(default)]
    pub capabilities: AgentCapabilities,
    #[serde(default)]
    pub planning_mode: PlanningMode,
}

impl AgentDefinition {
    pub fn minimal_voice_agent(
        id: impl Into<String>,
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            system_prompt: system_prompt.into(),
            default_model: default_model.into(),
            capabilities: AgentCapabilities::default(),
            planning_mode: PlanningMode::Auto,
        }
    }
}
