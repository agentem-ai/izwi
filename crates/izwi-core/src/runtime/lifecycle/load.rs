use crate::error::Result;
use crate::model::ModelVariant;
use crate::runtime::service::RuntimeService;

impl RuntimeService {
    /// Load a model for inference.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;
        let instantiated = self.instantiate_model(acquired).await?;
        self.publish_loaded_model(instantiated).await
    }
}
