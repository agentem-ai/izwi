use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{ipc::Channel, AppHandle, State};
use tauri_plugin_updater::{Update, UpdaterExt};
use url::Url;

use super::updater_contract::{
    github_manifest_url, github_releases_api_url, install_behavior_for_platform,
    parse_beta_sequence, PlatformInstallBehavior, UpdaterContract,
};

const DEFAULT_REQUEST_TIMEOUT_SECONDS: u64 = 20;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateMetadata {
    pub version: String,
    pub current_version: String,
    pub notes: Option<String>,
    pub published_at: Option<String>,
    pub release_tag: String,
    pub manifest_url: String,
    pub platform_behavior: PlatformInstallBehavior,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", content = "data")]
pub enum DownloadEvent {
    #[serde(rename_all = "camelCase")]
    Started {
        content_length: Option<u64>,
    },
    #[serde(rename_all = "camelCase")]
    Progress {
        chunk_length: usize,
        content_length: Option<u64>,
    },
    Finished,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallResult {
    pub app_exits_during_install: bool,
    pub supports_restart_later: bool,
}

#[derive(Default)]
pub struct UpdaterState {
    pending_update: Mutex<Option<Update>>,
}

impl UpdaterState {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum UpdaterError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Updater(#[from] tauri_plugin_updater::Error),
    #[error(transparent)]
    Url(#[from] url::ParseError),
    #[error("no matching beta release found for channel")]
    NoMatchingBetaRelease,
    #[error("there is no pending update")]
    NoPendingUpdate,
}

impl Serialize for UpdaterError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

type Result<T> = std::result::Result<T, UpdaterError>;

#[derive(Debug, Clone, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    draft: bool,
    assets: Vec<GitHubReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize)]
struct GitHubReleaseAsset {
    name: String,
}

#[derive(Debug, Clone)]
struct ResolvedRelease {
    tag_name: String,
    manifest_url: Url,
}

#[tauri::command]
pub async fn check_for_beta_update(
    app: AppHandle,
    updater_state: State<'_, UpdaterState>,
) -> Result<Option<UpdateMetadata>> {
    let contract = UpdaterContract::default();
    let release = resolve_latest_release(&contract).await?;
    let updater = app
        .updater_builder()
        .timeout(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        .endpoints(vec![release.manifest_url.clone()])?
        .build()?;
    let update = updater.check().await?;

    let update_metadata = update.as_ref().map(|update| {
        let platform_behavior = install_behavior_for_platform(std::env::consts::OS);
        UpdateMetadata {
            version: update.version.clone(),
            current_version: update.current_version.clone(),
            notes: update.body.clone(),
            published_at: update.date.map(|date| date.to_string()),
            release_tag: release.tag_name.clone(),
            manifest_url: release.manifest_url.to_string(),
            platform_behavior,
        }
    });

    let mut pending_slot = updater_state
        .pending_update
        .lock()
        .expect("poisoned updater state mutex");
    *pending_slot = update;

    Ok(update_metadata)
}

#[tauri::command]
pub async fn install_beta_update(
    updater_state: State<'_, UpdaterState>,
    on_event: Channel<DownloadEvent>,
) -> Result<InstallResult> {
    let update = {
        let mut pending_slot = updater_state
            .pending_update
            .lock()
            .expect("poisoned updater state mutex");
        pending_slot.take().ok_or(UpdaterError::NoPendingUpdate)?
    };

    let _ = on_event.send(DownloadEvent::Started {
        content_length: None,
    });

    update
        .download_and_install(
            |chunk_length, content_length| {
                let _ = on_event.send(DownloadEvent::Progress {
                    chunk_length,
                    content_length,
                });
            },
            || {
                let _ = on_event.send(DownloadEvent::Finished);
            },
        )
        .await?;

    let behavior = install_behavior_for_platform(std::env::consts::OS);

    Ok(InstallResult {
        app_exits_during_install: behavior.app_exits_during_install,
        supports_restart_later: behavior.supports_restart_later,
    })
}

#[tauri::command]
pub fn relaunch_after_update(app: AppHandle) {
    app.restart();
}

async fn resolve_latest_release(contract: &UpdaterContract) -> Result<ResolvedRelease> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECONDS))
        .build()?;
    let endpoint = github_releases_api_url(contract);

    let releases = client
        .get(endpoint)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .header(
            reqwest::header::USER_AGENT,
            format!("izwi-desktop/{}", env!("CARGO_PKG_VERSION")),
        )
        .send()
        .await?
        .error_for_status()?
        .json::<Vec<GitHubRelease>>()
        .await?;

    let best_release =
        select_best_release(&releases, contract).ok_or(UpdaterError::NoMatchingBetaRelease)?;

    let manifest_url = Url::parse(&github_manifest_url(contract, &best_release.tag_name))?;
    Ok(ResolvedRelease {
        tag_name: best_release.tag_name.clone(),
        manifest_url,
    })
}

fn release_has_manifest_asset(release: &GitHubRelease, contract: &UpdaterContract) -> bool {
    release
        .assets
        .iter()
        .any(|asset| asset.name == contract.manifest_asset_name)
}

fn select_best_release<'a>(
    releases: &'a [GitHubRelease],
    contract: &UpdaterContract,
) -> Option<&'a GitHubRelease> {
    releases
        .iter()
        .filter(|release| !release.draft)
        .filter(|release| release_has_manifest_asset(release, contract))
        .filter_map(|release| {
            parse_beta_sequence(contract.channel, &release.tag_name)
                .map(|sequence| (sequence, release))
        })
        .max_by_key(|(sequence, _)| *sequence)
        .map(|(_, release)| release)
}

#[cfg(test)]
mod tests {
    use super::{
        release_has_manifest_asset, select_best_release, GitHubRelease, GitHubReleaseAsset,
    };
    use crate::app::updater_contract::UpdaterContract;

    #[test]
    fn detects_manifest_asset_on_release() {
        let contract = UpdaterContract::default();
        let release = GitHubRelease {
            tag_name: "v0.1.0-beta-10".to_string(),
            draft: false,
            assets: vec![
                GitHubReleaseAsset {
                    name: "izwi_linux_x86_64.AppImage".to_string(),
                },
                GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                },
            ],
        };

        assert!(release_has_manifest_asset(&release, &contract));
    }

    #[test]
    fn selects_highest_beta_release_with_manifest() {
        let contract = UpdaterContract::default();
        let releases = vec![
            GitHubRelease {
                tag_name: "v0.1.0-beta-2".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                }],
            },
            GitHubRelease {
                tag_name: "v0.1.0-beta-11".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "latest-beta.json".to_string(),
                }],
            },
            GitHubRelease {
                tag_name: "v0.1.0-beta-12".to_string(),
                draft: false,
                assets: vec![GitHubReleaseAsset {
                    name: "something-else.json".to_string(),
                }],
            },
        ];
        let selected = select_best_release(&releases, &contract);

        assert_eq!(
            selected.map(|release| release.tag_name.as_str()),
            Some("v0.1.0-beta-11")
        );
    }
}
