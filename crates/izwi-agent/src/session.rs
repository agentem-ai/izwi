use crate::planner::PlanSummary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub agent_id: String,
    pub thread_id: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnInput {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub input_summary: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStarted {
        session_id: String,
        thread_id: String,
    },
    PlanCreated {
        steps: Vec<String>,
    },
    ToolCallStarted {
        name: String,
    },
    ToolCallCompleted {
        name: String,
        output: String,
    },
    AssistantMessage {
        content: String,
    },
    TurnCompleted {
        session_id: String,
        thread_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTurnResult {
    pub assistant_text: String,
    pub model_id: String,
    pub plan: Option<PlanSummary>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub events: Vec<AgentEvent>,
}
