use crate::errors::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    pub session_id: String,
    pub thread_id: String,
    pub user_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub text: String,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;

    fn description(&self) -> &'static str;

    fn should_auto_run(&self, _user_text: &str) -> bool {
        false
    }

    async fn invoke(&self, ctx: ToolContext) -> Result<ToolOutput>;
}

#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.push(Arc::new(tool));
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tools
            .iter()
            .map(|tool| tool.name().to_string())
            .collect()
    }

    pub fn find_auto_tool(&self, user_text: &str) -> Option<Arc<dyn Tool>> {
        self.tools
            .iter()
            .find(|tool| tool.should_auto_run(user_text))
            .cloned()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTool;

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn description(&self) -> &'static str {
        "A no-op tool used for testing and scaffolding."
    }

    async fn invoke(&self, _ctx: ToolContext) -> Result<ToolOutput> {
        Ok(ToolOutput {
            text: "No-op tool executed.".to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &'static str {
        "time"
    }

    fn description(&self) -> &'static str {
        "Returns the current UNIX time in seconds."
    }

    fn should_auto_run(&self, user_text: &str) -> bool {
        let lower = user_text.to_lowercase();
        (lower.contains("time") || lower.contains("date")) && !lower.contains("times")
    }

    async fn invoke(&self, _ctx: ToolContext) -> Result<ToolOutput> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Ok(ToolOutput {
            text: format!("Current UNIX timestamp (seconds): {now}"),
        })
    }
}
