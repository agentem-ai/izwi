use crate::Shell;
use clap::CommandFactory;
use clap_complete::generate;
use std::io;

pub fn execute(shell: Shell) {
    let mut cmd = crate::Cli::command();

    match shell {
        Shell::Bash => {
            generate(
                clap_complete::shells::Bash,
                &mut cmd,
                "izwi",
                &mut io::stdout(),
            );
        }
        Shell::Zsh => {
            generate(
                clap_complete::shells::Zsh,
                &mut cmd,
                "izwi",
                &mut io::stdout(),
            );
        }
        Shell::Fish => {
            generate(
                clap_complete::shells::Fish,
                &mut cmd,
                "izwi",
                &mut io::stdout(),
            );
        }
        Shell::PowerShell => {
            generate(
                clap_complete::shells::PowerShell,
                &mut cmd,
                "izwi",
                &mut io::stdout(),
            );
        }
        Shell::Elvish => {
            generate(
                clap_complete::shells::Elvish,
                &mut cmd,
                "izwi",
                &mut io::stdout(),
            );
        }
    }
}
