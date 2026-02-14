use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::BenchCommands;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct RuntimeTelemetrySnapshot {
    requests_queued: u64,
    requests_completed: u64,
    requests_failed: u64,
    requests_active: u64,
    worker_restarts: u64,
    worker_panics: u64,
    queue_wait_ms_avg: f64,
    queue_wait_ms_p50: f64,
    queue_wait_ms_p95: f64,
    prefill_ms_avg: f64,
    prefill_ms_p50: f64,
    prefill_ms_p95: f64,
    decode_ms_avg: f64,
    decode_ms_p50: f64,
    decode_ms_p95: f64,
    end_to_end_ms_avg: f64,
    end_to_end_ms_p50: f64,
    end_to_end_ms_p95: f64,
}

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
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }

    theme.step(1, 3, &format!("Benchmarking TTS with '{}'", model));
    let metrics_before = fetch_runtime_metrics(server).await;

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
    print_runtime_delta(metrics_before, fetch_runtime_metrics(server).await);

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
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }

    theme.step(1, 3, &format!("Benchmarking ASR with '{}'", model));
    let metrics_before = fetch_runtime_metrics(server).await;

    // Use sample audio if no file provided
    let audio_file = file.unwrap_or_else(|| std::path::PathBuf::from("data/test.wav"));

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
    print_runtime_delta(metrics_before, fetch_runtime_metrics(server).await);

    Ok(())
}

async fn bench_throughput(
    server: &str,
    duration: u64,
    concurrent: u32,
    theme: &Theme,
) -> Result<()> {
    if duration == 0 {
        return Err(CliError::InvalidInput(
            "Duration must be greater than 0 seconds".to_string(),
        ));
    }
    if concurrent == 0 {
        return Err(CliError::InvalidInput(
            "Concurrent requests must be greater than 0".to_string(),
        ));
    }

    theme.step(
        1,
        1,
        &format!("Throughput test: {}s, {} concurrent", duration, concurrent),
    );

    println!("Running throughput benchmark against /v1/health...");
    let client = http::client(Some(std::time::Duration::from_secs(5)))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(duration);

    let mut workers = Vec::new();
    for _ in 0..concurrent {
        let client = client.clone();
        let server = server.to_string();
        workers.push(tokio::spawn(async move {
            let mut success = 0u64;
            let mut failed = 0u64;
            while std::time::Instant::now() < deadline {
                match client.get(format!("{}/v1/health", server)).send().await {
                    Ok(resp) if resp.status().is_success() => success += 1,
                    _ => failed += 1,
                }
            }
            (success, failed)
        }));
    }

    let mut success = 0u64;
    let mut failed = 0u64;
    for worker in workers {
        let (ok, err) = worker
            .await
            .map_err(|e| CliError::Other(format!("Benchmark worker failed: {}", e)))?;
        success += ok;
        failed += err;
    }

    let total = success + failed;
    let rps = total as f64 / duration as f64;
    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Successful: {:.0}", success);
    println!("  Failed:     {:.0}", failed);
    println!("  Total:      {:.0}", total);
    println!("  Throughput: {:.2} req/s", rps);

    Ok(())
}

async fn run_tts_request(server: &str, model: &str, text: &str) -> Result<()> {
    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
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
    let audio_base64 = STANDARD.encode(&audio_data);

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
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
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let index = (p * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}

async fn fetch_runtime_metrics(server: &str) -> Option<RuntimeTelemetrySnapshot> {
    let client = http::client(Some(std::time::Duration::from_secs(3))).ok()?;
    let response = client
        .get(format!("{}/internal/metrics", server.trim_end_matches('/')))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json::<RuntimeTelemetrySnapshot>().await.ok()
}

fn print_runtime_delta(
    before: Option<RuntimeTelemetrySnapshot>,
    after: Option<RuntimeTelemetrySnapshot>,
) {
    let (Some(before), Some(after)) = (before, after) else {
        println!("\nRuntime telemetry delta: unavailable (/internal/metrics not reachable)");
        return;
    };

    let completed_delta = after
        .requests_completed
        .saturating_sub(before.requests_completed);
    let failed_delta = after.requests_failed.saturating_sub(before.requests_failed);
    let queued_delta = after.requests_queued.saturating_sub(before.requests_queued);
    let restart_delta = after.worker_restarts.saturating_sub(before.worker_restarts);
    let panic_delta = after.worker_panics.saturating_sub(before.worker_panics);

    println!(
        "\n{}",
        console::style("Runtime Telemetry Delta:")
            .bold()
            .underlined()
    );
    println!("  Queued:             {}", queued_delta);
    println!("  Completed:          {}", completed_delta);
    println!("  Failed:             {}", failed_delta);
    println!("  Active (current):   {}", after.requests_active);
    println!("  Worker restarts:    {}", restart_delta);
    println!("  Worker panics:      {}", panic_delta);
    println!(
        "  Queue wait (avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
        after.queue_wait_ms_avg, after.queue_wait_ms_p50, after.queue_wait_ms_p95
    );
    println!(
        "  Prefill (avg/p50/p95):    {:.2} / {:.2} / {:.2} ms",
        after.prefill_ms_avg, after.prefill_ms_p50, after.prefill_ms_p95
    );
    println!(
        "  Decode (avg/p50/p95):     {:.2} / {:.2} / {:.2} ms",
        after.decode_ms_avg, after.decode_ms_p50, after.decode_ms_p95
    );
    println!(
        "  End-to-end (avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
        after.end_to_end_ms_avg, after.end_to_end_ms_p50, after.end_to_end_ms_p95
    );
}
