use crate::error::{CliError, Result};
use crate::style::Theme;
use console::style;

pub struct ChatArgs {
    pub model: String,
    pub system: Option<String>,
    pub voice: Option<String>,
}

pub async fn execute(args: ChatArgs, server: &str, theme: &Theme) -> Result<()> {
    theme.print_banner();
    
    println!("{}", style(format!("Chat mode with '{}'", args.model)).bold());
    println!("Type your message and press Enter. Use /quit or /exit to quit.\n");

    let system_msg = args.system.as_deref().unwrap_or(
        "You are a helpful AI assistant with voice capabilities."
    );

    let mut messages = vec![
        serde_json::json!({
            "role": "system",
            "content": system_msg
        })
    ];

    loop {
        // Get user input
        let input = dialoguer::Input::<String>::new()
            .with_prompt("You")
            .interact_text()
            .map_err(|e| CliError::Other(e.to_string()))?;

        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }

        // Check for quit commands
        if input == "/quit" || input == "/exit" || input == "/q" {
            println!("Goodbye!");
            break;
        }

        if input == "/help" {
            println!("Commands:");
            println!("  /quit, /exit, /q - Exit chat");
            println!("  /help - Show this help");
            println!("  /clear - Clear conversation history");
            continue;
        }

        if input == "/clear" {
            messages.clear();
            messages.push(serde_json::json!({
                "role": "system",
                "content": system_msg
            }));
            println!("Conversation history cleared.");
            continue;
        }

        // Add user message
        messages.push(serde_json::json!({
            "role": "user",
            "content": input
        }));

        // Send request to chat completions endpoint
        let request_body = serde_json::json!({
            "model": args.model,
            "messages": messages,
            "stream": false,
        });

        let client = reqwest::Client::new();
        let response = client
            .post(format!("{}/v1/chat/completions", server))
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CliError::ConnectionError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(CliError::ApiError { status, message: text });
        }

        let result: serde_json::Value = response.json().await?;
        
        // Extract assistant's response
        if let Some(choices) = result.get("choices").and_then(|c| c.as_array()) {
            if let Some(first) = choices.first() {
                if let Some(content) = first
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                {
                    println!("{}: {}", style("Assistant").green().bold(), content);
                    
                    // Add to message history
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content
                    }));
                }
            }
        }

        println!();
    }

    Ok(())
}
