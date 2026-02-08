use console::style;

pub const HELP_TEMPLATE: &str = r#"
{before-help}{name} {version}
{author-with-newline}{about-with-newline}
{usage-heading}
{tab}{usage}

{all-args}{after-help}
"#;

pub const BANNER: &str = r#"
 ___ ____ ____ 
|_ _|_  /_  /  High-performance audio inference
 | | / / / /   Text-to-Speech & Speech-to-Text
|___/___/___|  Optimized for Apple Silicon & CUDA
"#;

/// Theme for styled terminal output
#[derive(Clone)]
pub struct Theme {
    pub accent: fn(&str) -> console::StyledObject<&str>,
    pub success: fn(&str) -> console::StyledObject<&str>,
    pub error: fn(&str) -> console::StyledObject<&str>,
    pub warning: fn(&str) -> console::StyledObject<&str>,
    pub info: fn(&str) -> console::StyledObject<&str>,
    pub muted: fn(&str) -> console::StyledObject<&str>,
    pub bold: fn(&str) -> console::StyledObject<&str>,
    pub no_color: bool,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            accent: |s| style(s).cyan().bold(),
            success: |s| style(s).green().bold(),
            error: |s| style(s).red().bold(),
            warning: |s| style(s).yellow(),
            info: |s| style(s).blue(),
            muted: |s| style(s).dim(),
            bold: |s| style(s).bold(),
            no_color: false,
        }
    }
}

impl Theme {
    pub fn no_color() -> Self {
        Self {
            accent: |s| style(s),
            success: |s| style(s),
            error: |s| style(s),
            warning: |s| style(s),
            info: |s| style(s),
            muted: |s| style(s),
            bold: |s| style(s),
            no_color: true,
        }
    }

    pub fn print_banner(&self) {
        if self.no_color {
            println!("{}", BANNER.trim_start_matches('\n'));
        } else {
            println!("{}", (self.accent)(BANNER.trim_start_matches('\n')));
        }
    }

    pub fn success(&self, msg: &str) {
        println!("{} {}", (self.success)("✓"), msg);
    }

    pub fn error(&self, msg: &str) {
        eprintln!("{} {}", (self.error)("✗"), msg);
    }

    pub fn warning(&self, msg: &str) {
        println!("{} {}", (self.warning)("⚠"), msg);
    }

    pub fn info(&self, msg: &str) {
        println!("{} {}", (self.info)("ℹ"), msg);
    }

    pub fn step(&self, n: usize, total: usize, msg: &str) {
        println!("{} {}", (self.accent)(&format!("[{}/{}]", n, total)), msg);
    }
}

/// Progress bar styles
pub fn progress_bar_style() -> indicatif::ProgressStyle {
    indicatif::ProgressStyle::default_bar()
        .template("{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .unwrap()
        .progress_chars("#>-")
}

pub fn spinner_style() -> indicatif::ProgressStyle {
    indicatif::ProgressStyle::default_spinner()
        .template("{spinner:.cyan} {msg}")
        .unwrap()
}
