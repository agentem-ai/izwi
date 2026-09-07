use super::*;
use crate::engine::{ExecutionAdapterBinding, StageDescriptor, StageWorkSelector};
use crate::models::architectures::fish_s2::{
    codec::decode_workspace_bytes, FishS2GenerationParams, FishS2PreparedArtifact, FishS2TtsModel,
};
use crate::models::registry::FishS2TtsModelLease;
use crate::runtime::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};

fn prepared_request(frames: usize) -> EngineCoreRequest {
    let mut request = EngineCoreRequest::tts("Tell me something funny, I stay laughing")
        .with_model_variant(ModelVariant::FishAudioS2Pro);
    request
        .install_fish_s2_tts_execution_model(
            ModelVariant::FishAudioS2Pro,
            FishS2TtsModelLease::for_test(FishS2TtsModel::for_test()),
            FishS2PreparedArtifact::test_prompt(11, 32),
            FishS2GenerationParams {
                max_frames: frames,
                ..Default::default()
            },
            8192,
        )
        .unwrap();
    request.validate_execution_preparation().unwrap();
    request
}

fn loaded_binding(backend: BackendKind) -> (ExecutionAdapterBinding, StageDescriptor) {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::FishAudioS2Pro,
        backend,
    )
    .unwrap();
    let contract = draft
        .execution_contracts(CapabilityKind::Tts)
        .unwrap()
        .into_iter()
        .find(|contract| contract.execution_profile.mode == ExecutionMode::Sequence)
        .unwrap();
    let stage = contract
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceFinalize)
        .unwrap()
        .clone();
    (contract.adapter_binding().unwrap(), stage)
}

fn finalize_cost(
    request: &EngineCoreRequest,
    stage: Option<&StageDescriptor>,
    backend: BackendKind,
) -> Result<WorkCost> {
    EngineCore::work_cost(
        request,
        &WorkUnit::SequenceFinalize {
            max_output_steps: 1,
        },
        stage,
        backend,
    )
}

#[test]
fn fish_s2_preparation_binds_then_prices_scalar_codec_on_each_backend() {
    for backend in [BackendKind::Cpu, BackendKind::Cuda, BackendKind::Metal] {
        for frames in [1, 512, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES] {
            let mut request = prepared_request(frames);
            let (binding, stage) = loaded_binding(backend);
            assert_eq!(stage.batch_mode, NativeBatchMode::None);
            assert!(request.execution_adapter_binding().is_none());
            assert!(request.prepared_stage_cost(stage.id).is_none());
            request.bind_execution_adapter(binding).unwrap();
            request.validate_execution_preparation().unwrap();
            let cost = finalize_cost(&request, Some(&stage), backend).unwrap();
            let bytes = decode_workspace_bytes(frames).unwrap();
            let mut expected = ResourceVector::zero();
            match backend {
                BackendKind::Cpu => expected.host_bytes = ResourceAmount::Known(bytes),
                BackendKind::Cuda => expected.device_bytes = ResourceAmount::Known(bytes),
                BackendKind::Metal => expected.unified_bytes = ResourceAmount::Known(bytes),
            }
            assert_eq!(cost, WorkCost::with_workspace(1, 1, expected));
            assert!(bytes <= stage.max_workspace_bytes);
            assert!(request.prepared_stage_cost(stage.id).is_none());
        }
    }
}

#[test]
fn fish_s2_codec_cost_rejects_missing_binding_and_foreign_stage() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cuda);
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
    request.bind_execution_adapter(binding.clone()).unwrap();
    assert!(finalize_cost(&request, None, BackendKind::Cuda).is_err());
    let foreign_stage = binding
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
        .unwrap();
    assert!(finalize_cost(&request, Some(foreign_stage), BackendKind::Cuda).is_err());
    let mut changed_stage = stage.clone();
    changed_stage.name.push_str(".foreign");
    assert!(finalize_cost(&request, Some(&changed_stage), BackendKind::Cuda).is_err());
}

#[test]
fn fish_s2_codec_cost_rejects_stale_prepared_geometry() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cuda);
    request.bind_execution_adapter(binding).unwrap();
    request.params.max_tokens = 1;
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
}

#[test]
fn fish_s2_codec_cost_cannot_exceed_loaded_stage_ceiling() {
    let mut request = prepared_request(512);
    let (mut binding, mut stage) = loaded_binding(BackendKind::Cuda);
    stage.max_workspace_bytes = 1;
    binding.stages = binding
        .stages
        .iter()
        .map(|entry| {
            if entry.id == stage.id {
                stage.clone()
            } else {
                entry.clone()
            }
        })
        .collect::<Vec<_>>()
        .into();
    request.bind_execution_adapter(binding).unwrap();
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
}

#[test]
fn fish_s2_scalar_codec_remains_invalid_as_a_prepared_tensor_stage() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cpu);
    request.bind_execution_adapter(binding).unwrap();
    let error = request
        .install_prepared_stage_cost(stage.id, WorkCost::new(1, 1, 1))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("invalid prepared tensor-stage cost"));
}

#[test]
fn fish_s2_codec_cost_rejects_unsupported_frame_and_work_bounds() {
    let (binding, stage) = loaded_binding(BackendKind::Cpu);
    let mut request = prepared_request(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES + 1);
    request.bind_execution_adapter(binding.clone()).unwrap();
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cpu).is_err());
    let mut request = prepared_request(128);
    request.bind_execution_adapter(binding).unwrap();
    assert!(EngineCore::work_cost(
        &request,
        &WorkUnit::SequenceFinalize {
            max_output_steps: 2
        },
        Some(&stage),
        BackendKind::Cpu,
    )
    .is_err());
}

#[test]
fn fish_s2_decode_and_other_models_keep_generic_work_cost() {
    let mut request = prepared_request(128);
    let (binding, _) = loaded_binding(BackendKind::Cpu);
    let stage = binding
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
        .unwrap()
        .clone();
    request.bind_execution_adapter(binding).unwrap();
    let work = WorkUnit::SequenceStep {
        phase: SequencePhase::Decode,
        input: crate::engine::InputRange::new(0, 1).unwrap(),
        max_output_steps: 1,
        auxiliary_state: None,
    };
    let cost = EngineCore::work_cost(&request, &work, Some(&stage), BackendKind::Cpu).unwrap();
    assert_eq!(
        cost,
        WorkCost::new(
            1,
            1,
            stage.workspace_per_row_bytes + stage.workspace_per_work_unit_bytes
        )
    );
    let request = EngineCoreRequest::tts("Other model").with_model_variant(ModelVariant::Kokoro82M);
    let cost = finalize_cost(&request, None, BackendKind::Cpu).unwrap();
    assert_eq!(cost, WorkCost::new(1, 1, 0));
}
