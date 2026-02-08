use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::BenchCommands;
use base64;
use indicatif::{ProgressBar, ProgressStyle};

pub async fn execute(command: BenchCommands, server: &str, theme: &Theme) -> Result<()> {
    match command {
        BenchCommands::Tts {
            model,
            iterations,
            text,
            warmup,
        } => bench_tts(server, &model, iterations, &text, warmup, theme).await,
        BenchCommands::Asr {
            model,
            iterations,
            file,
            warmup,
        } => bench_asr(server, &model, iterations, file, warmup, theme).await,
        BenchCommands::Throughput {
            duration,
            concurrent,
        } => bench_throughput(server, duration, concurrent, theme).await,
    }
}

async fn bench_tts(
    server: &str,
    model: &str,
    iterations: u32,
    text: &str,
    warmup: bool,
    theme: &Theme,
) -> Result<()> {
    theme.step(1, 3, &format!("Benchmarking TTS with '{}'", model));

    if warmup {
        theme.info("Running warmup iteration...");
        let _ = run_tts_request(server, model, text).await?;
    }

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = run_tts_request(server, model, text).await?;
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);
        pb.inc(1);
    }

    pb.finish_with_message("Benchmark complete");

    // Calculate statistics
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0, f64::max);
    let p50 = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let p99 = percentile(&times, 0.99);

    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Iterations: {}", iterations);
    println!("  Average:    {:.2} ms", avg);
    println!("  Min:        {:.2} ms", min);
    println!("  Max:        {:.2} ms", max);
    println!("  P50:        {:.2} ms", p50);
    println!("  P95:        {:.2} ms", p95);
    println!("  P99:        {:.2} ms", p99);
    println!("  Throughput: {:.2} req/s", 1000.0 / avg);

    Ok(())
}

async fn bench_asr(
    server: &str,
    model: &str,
    iterations: u32,
    file: Option<std::path::PathBuf>,
    warmup: bool,
    theme: &Theme,
) -> Result<()> {
    theme.step(1, 3, &format!("Benchmarking ASR with '{}'", model));

    // Use sample audio if no file provided
    let audio_file = file.unwrap_or_else(|| std::path::PathBuf::from("test.wav"));

    if !audio_file.exists() {
        return Err(CliError::InvalidInput(format!(
            "Audio file not found: {}",
            audio_file.display()
        )));
    }

    if warmup {
        theme.info("Running warmup iteration...");
        let _ = run_asr_request(server, model, &audio_file).await?;
    }

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = run_asr_request(server, model, &audio_file).await?;
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);
        pb.inc(1);
    }

    pb.finish_with_message("Benchmark complete");

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Average:    {:.2} ms", avg);
    println!("  Throughput: {:.2} req/s", 1000.0 / avg);

    Ok(())
}

async fn bench_throughput(
    server: &str,
    duration: u64,
    concurrent: u32,
    theme: &Theme,
) -> Result<()> {
    theme.step(
        1,
        1,
        &format!("Throughput test: {}s, {} concurrent", duration, concurrent),
    );

    println!("Running throughput benchmark...");
    theme.info("Throughput benchmarking would run concurrent requests here");

    Ok(())
}

async fn run_tts_request(server: &str, model: &str, text: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let request_body = serde_json::json!({
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": "wav",
    });

    let response = client
        .post(format!("{}/v1/audio/speech", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    Ok(())
}

async fn run_asr_request(server: &str, model: &str, file: &std::path::PathBuf) -> Result<()> {
    let audio_data = tokio::fs::read(file).await.map_err(|e| CliError::Io(e))?;
    let audio_base64 = base64::encode(&audio_data);

    let client = reqwest::Client::new();
    let request_body = serde_json::json!({
        "model": model,
        "file": format!("data:audio/wav;base64,{}", audio_base64),
    });

    let response = client
        .post(format!("{}/v1/audio/transcriptions", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    Ok(())
}

fn percentile(data: &[f64], p: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = (p * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}
