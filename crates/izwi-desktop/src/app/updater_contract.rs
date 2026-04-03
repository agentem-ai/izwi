#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateChannel {
    Beta010,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdaterContract {
    pub channel: UpdateChannel,
    pub release_owner: String,
    pub release_repo: String,
    pub manifest_asset_name: String,
    pub releases_per_page: usize,
}

impl Default for UpdaterContract {
    fn default() -> Self {
        Self {
            channel: UpdateChannel::Beta010,
            release_owner: "izwi-ai".to_string(),
            release_repo: "izwi".to_string(),
            manifest_asset_name: "latest-beta.json".to_string(),
            releases_per_page: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformInstallBehavior {
    pub app_exits_during_install: bool,
    pub supports_restart_later: bool,
}

impl UpdateChannel {
    pub fn tag_prefix(self) -> &'static str {
        match self {
            UpdateChannel::Beta010 => "v0.1.0-beta-",
        }
    }
}

pub fn parse_beta_sequence(channel: UpdateChannel, tag_name: &str) -> Option<u64> {
    let suffix = tag_name.strip_prefix(channel.tag_prefix())?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse::<u64>().ok()
}

pub fn is_supported_beta_tag(channel: UpdateChannel, tag_name: &str) -> bool {
    parse_beta_sequence(channel, tag_name).is_some()
}

pub fn select_latest_beta_tag<'a, I>(channel: UpdateChannel, tag_names: I) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    tag_names
        .into_iter()
        .filter_map(|tag| parse_beta_sequence(channel, tag).map(|sequence| (sequence, tag)))
        .max_by_key(|(sequence, _)| *sequence)
        .map(|(_, tag)| tag)
}

pub fn github_releases_api_url(contract: &UpdaterContract) -> String {
    format!(
        "https://api.github.com/repos/{}/{}/releases?per_page={}",
        contract.release_owner, contract.release_repo, contract.releases_per_page
    )
}

pub fn github_manifest_url(contract: &UpdaterContract, tag_name: &str) -> String {
    format!(
        "https://github.com/{}/{}/releases/download/{}/{}",
        contract.release_owner, contract.release_repo, tag_name, contract.manifest_asset_name
    )
}

pub fn install_behavior_for_platform(target_os: &str) -> PlatformInstallBehavior {
    match target_os {
        "windows" => PlatformInstallBehavior {
            app_exits_during_install: true,
            supports_restart_later: false,
        },
        "macos" | "linux" => PlatformInstallBehavior {
            app_exits_during_install: false,
            supports_restart_later: true,
        },
        _ => PlatformInstallBehavior {
            app_exits_during_install: false,
            supports_restart_later: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        UpdateChannel, UpdaterContract, github_manifest_url, github_releases_api_url,
        install_behavior_for_platform, is_supported_beta_tag, parse_beta_sequence,
        select_latest_beta_tag,
    };

    #[test]
    fn parses_beta_sequence_with_valid_tag() {
        let channel = UpdateChannel::Beta010;
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta-10"), Some(10));
    }

    #[test]
    fn rejects_invalid_beta_tags() {
        let channel = UpdateChannel::Beta010;
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta"), None);
        assert_eq!(parse_beta_sequence(channel, "v0.1.1-beta-1"), None);
        assert_eq!(parse_beta_sequence(channel, "v0.1.0-beta-1a"), None);
        assert_eq!(parse_beta_sequence(channel, "0.1.0-beta-1"), None);
    }

    #[test]
    fn detects_supported_beta_tags() {
        let channel = UpdateChannel::Beta010;
        assert!(is_supported_beta_tag(channel, "v0.1.0-beta-1"));
        assert!(!is_supported_beta_tag(channel, "v0.1.0-alpha-1"));
    }

    #[test]
    fn selects_latest_beta_sequence() {
        let channel = UpdateChannel::Beta010;
        let tags = [
            "v0.1.0-beta-1",
            "v0.1.0-beta-10",
            "v0.1.0-alpha-9",
            "v0.1.0-beta-7",
        ];
        assert_eq!(
            select_latest_beta_tag(channel, tags.iter().copied()),
            Some("v0.1.0-beta-10")
        );
    }

    #[test]
    fn builds_github_endpoints_from_contract() {
        let contract = UpdaterContract::default();
        assert_eq!(
            github_releases_api_url(&contract),
            "https://api.github.com/repos/izwi-ai/izwi/releases?per_page=50"
        );
        assert_eq!(
            github_manifest_url(&contract, "v0.1.0-beta-10"),
            "https://github.com/izwi-ai/izwi/releases/download/v0.1.0-beta-10/latest-beta.json"
        );
    }

    #[test]
    fn exposes_platform_install_contract() {
        let windows = install_behavior_for_platform("windows");
        assert!(windows.app_exits_during_install);
        assert!(!windows.supports_restart_later);

        let macos = install_behavior_for_platform("macos");
        assert!(!macos.app_exits_during_install);
        assert!(macos.supports_restart_later);

        let linux = install_behavior_for_platform("linux");
        assert!(!linux.app_exits_during_install);
        assert!(linux.supports_restart_later);
    }
}
