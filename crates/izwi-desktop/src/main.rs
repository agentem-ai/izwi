use anyhow::Result;
use clap::Parser;

mod app;

#[tokio::main]
async fn main() -> Result<()> {
    app::run(app::DesktopArgs::parse())
}
