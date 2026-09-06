//! Exercise numerical draft recovery through real tiny target/MTP forwards.
use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
use crate::models::architectures::qwen38::mtp::tests::{
    tiny_config, write_tiny_checkpoint, TestDir,
};
use crate::models::architectures::qwen38::native::IndexedSafetensors;
use candle_core::Device;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::BTreeMap;
use std::sync::Arc;

fn model() -> Qwen38ChatModel {
    let dir = TestDir::new("chat-recovery");
    let native = tiny_config();
    write_tiny_checkpoint(dir.path(), &native);
    // Reuse the zero-projection MTP transformer as the tiny target. Distinct
    // embeddings and an identity-like output head give non-uniform logits.
    let bytes = fs::read(dir.path().join("mtp.safetensors")).unwrap();
    let mtp = SafeTensors::deserialize(&bytes).unwrap();
    let mut views = BTreeMap::new();
    for name in mtp.names() {
        if name.starts_with("mtp.layers.0.") {
            views.insert(
                name.replacen("mtp.layers.0.", "model.language_model.layers.0.", 1),
                mtp.tensor(name).unwrap(),
            );
        }
    }
    let matrix: Vec<u8> = (0..32)
        .flat_map(|i| {
            let value = if i % 4 == (i / 4) % 4 { 1.0 } else { 0.125 };
            half::bf16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect();
    let norm = vec![0u8; 8];
    for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
        views.insert(
            name.into(),
            TensorView::new(Dtype::BF16, vec![8, 4], &matrix).unwrap(),
        );
    }
    views.insert(
        "model.language_model.norm.weight".into(),
        TensorView::new(Dtype::BF16, vec![4], &norm).unwrap(),
    );
    safetensors::serialize_to_file(&views, &None, &dir.path().join("target.safetensors")).unwrap();
    let index_path = dir.path().join("model.safetensors.index.json");
    let mut index: serde_json::Value =
        serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
    for name in views.keys() {
        index["weight_map"][name] = serde_json::json!("target.safetensors");
    }
    fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
    let tensors = IndexedSafetensors::open(dir.path()).unwrap();
    let mut performance = crate::performance::PerformanceConfig::default();
    performance.cuda.mtp_draft_tokens = 2;
    let text_model = Qwen38TextModel::load_native_with_performance(
        &tensors,
        &native,
        &Device::Cpu,
        ProjectionMaterialization::F32,
        &performance.cuda,
    )
    .unwrap();
    let inventory = tensors.validate_mtp_tensor_manifest(&native).unwrap();
    let mtp_head = Qwen38MtpHead::load_native_with_performance(
        &tensors,
        &native,
        &inventory,
        &Device::Cpu,
        ProjectionMaterialization::F32,
        &performance.cuda,
    )
    .unwrap();
    let inner = Tokenizer::from_hf_json_bytes(br#"{
        "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"unk_token":"a"}
    }"#).unwrap();
    Qwen38ChatModel {
        device_kind: BackendKind::Cpu,
        performance,
        load_timing: serde_json::json!({}),
        prefill_chunk_size: 4,
        cuda_compute_capability: None,
        kv_storage_provider: Qwen38KvStorageProvider::CpuF32,
        variant: ModelVariant::Qwen3827BFp8,
        tokenizer: Qwen38Tokenizer {
            inner,
            vocab_size: 8,
            specials: SpecialTokenIds {
                im_end: 100,
                eos: 101,
                eos_alt: None,
            },
            literal_special_tokens: Vec::new(),
            chat_template: String::new(),
            default_enable_thinking: false,
        },
        text_config: native.text,
        text_model,
        mtp_policy: Qwen38MtpPolicy::Enabled { draft_tokens: 2 },
        mtp_head: Some(mtp_head),
    }
}

fn cache(model_layer: u32) -> PhysicalPagedKvCache {
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(91),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(model_layer);
    let binding = KvLayerBinding {
        model_layer,
        physical_layer: 0,
    };
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 4,
            capacity_pages: 8,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .unwrap(),
    );
    let blocks = (0..8)
        .map(|index| CacheBlockRef {
            arena: id,
            group,
            index,
            slot_generation: 1,
        })
        .collect();
    PhysicalPagedKvCache::new(arena, vec![binding], blocks, 0).unwrap()
}

fn start(model: &Qwen38ChatModel, temperature: f32) -> ChatDecodeState {
    let config = ChatGenerationConfig {
        temperature,
        top_p: 0.9,
        top_k: 6,
        repetition_penalty: 1.1,
        presence_penalty: 0.1,
        seed: 42,
        ..Default::default()
    };
    let prepared = Qwen38PreparedPrompt {
        prompt_ids: vec![1, 2],
        prompt_positions: vec![[0; 3], [1; 3]],
        next_text_position: 2,
    };
    model
        .start_decode_state_physical(&[], 12, &config, Some(&prepared), cache(0), Some(cache(1)))
        .unwrap()
}

fn reservation(cache: &PhysicalPagedKvCache) -> PhysicalPagedKvCache {
    PhysicalPagedKvCache::new(
        cache.arena().clone(),
        vec![cache.layer_binding(0).unwrap()],
        cache.blocks.clone(),
        cache.context_len(),
    )
    .unwrap()
}

fn poison_mtp_cache(state: &ChatDecodeState) {
    use crate::backends::kv::KvWriteArgs;
    use crate::kv::KvSlotRef;
    let cache = state.mtp_physical_kv.as_ref().unwrap();
    let slots = cache
        .arena()
        .lower_slots(&[KvSlotRef {
            block: cache.blocks[0],
            offset: 0,
        }])
        .unwrap();
    let keys = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
    let values = Tensor::full(f32::NAN, (1, 1, 2), &Device::Cpu).unwrap();
    cache
        .arena()
        .write_slots(
            cache.layer_binding(0).unwrap(),
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots: slots.as_ref(),
            },
        )
        .unwrap()
        .wait()
        .unwrap();
}

#[test]
fn nonfinite_mtp_draft_recovers_to_exact_scalar_sequence() {
    let model = model();
    for (temperature, fail_second_draft) in [(0.0, false), (0.8, false), (0.0, true), (0.8, true)] {
        let mut actual = start(&model, temperature);
        let mut reference = start(&model, temperature);
        reference.adaptive_mtp.disable_after_nonfinite_draft();
        // Emit bootstrap and a healthy scalar step before the fault, matching
        // a stream that fails only after it has already produced valid output.
        for _ in 0..2 {
            model.decode_quantum(&mut actual, 1).unwrap();
            model.decode_quantum(&mut reference, 1).unwrap();
        }
        if fail_second_draft {
            // Proposal one uses a valid anchor and consumes a draft RNG draw.
            // The recurrent forward then reads poisoned V, so proposal two
            // fails after one provisional KV append.
            poison_mtp_cache(&actual);
        } else {
            actual.mtp_anchor_hidden =
                Some(Tensor::full(f32::NAN, (1, 1, 4), &Device::Cpu).unwrap());
        }
        let draft_rng = actual.draft_rng.state;
        let checkpoint = actual
            .begin_shared_step_quantum(
                reservation(&actual.physical_kv),
                actual.mtp_physical_kv.as_ref().map(reservation),
            )
            .unwrap();
        let reference_checkpoint = reference
            .begin_shared_step_quantum(
                reservation(&reference.physical_kv),
                reference.mtp_physical_kv.as_ref().map(reservation),
            )
            .unwrap();
        let recovered = model.decode_quantum(&mut actual, 4).unwrap();
        let expected = model.decode_quantum(&mut reference, 4).unwrap();
        // Check the immediate recovery output before cancellation can erase
        // evidence of an incorrect token, RNG draw or canonical history edit.
        assert_eq!(recovered.delta, expected.delta);
        assert_eq!(recovered.text, expected.text);
        assert_eq!(
            recovered.input_tokens_committed,
            expected.input_tokens_committed
        );
        assert_eq!(actual.history_ids, reference.history_ids);
        assert_eq!(actual.rng.state, reference.rng.state);
        assert_eq!(actual.pending_token, reference.pending_token);
        assert_eq!(actual.next_text_position, reference.next_text_position);
        assert_eq!(
            actual.physical_kv.context_len(),
            reference.physical_kv.context_len()
        );
        assert_eq!(
            actual.mtp_physical_kv.as_ref().unwrap().context_len(),
            actual.physical_kv.context_len()
        );
        assert!(actual.adaptive_mtp.speculation_disabled());
        assert_eq!(actual.draft_rng.state, draft_rng);
        // Cancellation rewinds output and RNG but must not erase the numerical
        // latch, even when the checkpoint's anchor was healthy.
        actual.rollback_shared_step_quantum(checkpoint);
        reference.rollback_shared_step_quantum(reference_checkpoint);
        assert!(actual.adaptive_mtp.speculation_disabled());
        while !actual.finished {
            let step = model.decode_quantum(&mut actual, 4).unwrap();
            let expected = model.decode_quantum(&mut reference, 4).unwrap();
            assert_eq!(step.delta, expected.delta);
            assert_eq!(step.input_tokens_committed, expected.input_tokens_committed);
            assert_eq!(actual.history_ids, reference.history_ids);
            assert_eq!(actual.rng.state, reference.rng.state);
            assert_eq!(actual.draft_rng.state, draft_rng);
            assert_eq!(actual.pending_token, reference.pending_token);
            assert_eq!(actual.next_text_position, reference.next_text_position);
            assert_eq!(
                actual.physical_kv.context_len(),
                reference.physical_kv.context_len()
            );
            assert_eq!(
                actual.mtp_physical_kv.as_ref().unwrap().context_len(),
                actual.physical_kv.context_len()
            );
            assert!(actual.adaptive_mtp.speculation_disabled());
            assert_eq!(actual.finished, reference.finished);
        }
        assert_eq!(actual.tokens_generated, 12);
        assert_eq!(actual.assembled, reference.assembled);
        // The latch belongs to the failed request, not the loaded model.
        assert!(!start(&model, temperature)
            .adaptive_mtp
            .speculation_disabled());
    }
}
