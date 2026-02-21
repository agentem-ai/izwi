use crate::model::ModelVariant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeModelFamily {
    Asr,
    Diarization,
    Chat,
    Lfm2,
    Voxtral,
    Tts,
    Auxiliary,
}

struct ModelFamilyRegistration {
    family: RuntimeModelFamily,
    matcher: fn(ModelVariant) -> bool,
}

fn is_asr_family(variant: ModelVariant) -> bool {
    variant.is_asr() || variant.is_forced_aligner()
}

fn is_diarization_family(variant: ModelVariant) -> bool {
    variant.is_diarization()
}

fn is_chat_family(variant: ModelVariant) -> bool {
    variant.is_chat()
}

fn is_lfm2_family(variant: ModelVariant) -> bool {
    variant.is_lfm2()
}

fn is_voxtral_family(variant: ModelVariant) -> bool {
    variant.is_voxtral()
}

fn is_tts_family(variant: ModelVariant) -> bool {
    variant.is_tts()
}

const MODEL_FAMILY_REGISTRY: &[ModelFamilyRegistration] = &[
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Asr,
        matcher: is_asr_family,
    },
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Diarization,
        matcher: is_diarization_family,
    },
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Chat,
        matcher: is_chat_family,
    },
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Lfm2,
        matcher: is_lfm2_family,
    },
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Voxtral,
        matcher: is_voxtral_family,
    },
    ModelFamilyRegistration {
        family: RuntimeModelFamily::Tts,
        matcher: is_tts_family,
    },
];

pub(super) fn resolve_runtime_model_family(variant: ModelVariant) -> RuntimeModelFamily {
    MODEL_FAMILY_REGISTRY
        .iter()
        .find(|registration| (registration.matcher)(variant))
        .map(|registration| registration.family)
        .unwrap_or(RuntimeModelFamily::Auxiliary)
}
