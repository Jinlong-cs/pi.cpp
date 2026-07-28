SHELL := /bin/bash

BUILD_DIR ?= build
BUILD_TYPE ?= Release
TRT_FLAGS ?= --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference
PI05_PREFIX_EMBED_TRT_FLAGS ?= $(TRT_FLAGS)
PI05_TRT_FLAGS ?= $(filter-out --fp16,$(TRT_FLAGS)) --stronglyTyped
DIT4DIT_ACTION_TRT_FLAGS ?= $(filter-out --fp16,$(TRT_FLAGS)) --stronglyTyped
DIT4DIT_FEATURE_TRT_FLAGS ?= --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference
GROOT_ACTION_TRT_FLAGS ?= $(filter-out --fp16,$(TRT_FLAGS)) --stronglyTyped
STARVLA_POLICY_TRT_FLAGS ?= --stronglyTyped --builderOptimizationLevel=3 --memPoolSize=workspace:8192 --skipInference
EVO1_BACKBONE_TRT_FLAGS ?= --stronglyTyped --builderOptimizationLevel=3 --memPoolSize=workspace:8192 --skipInference
EVO1_ACTION_TRT_FLAGS ?= --stronglyTyped --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference

PI_ONNX_DIR ?= assets/pi05_libero/onnx
PI_ENGINE_DIR ?= assets/pi05_libero/engines
PI05_INT8_ONNX_DIR ?= assets/pi05_libero_int8/onnx
PI05_INT8_ENGINE_DIR ?= assets/pi05_libero_int8/engines
PI05_INT8_TRT_FLAGS ?= --fp16 --int8 --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference

PREFIX_EMBED_ONNX ?= $(PI_ONNX_DIR)/prefix_embed.onnx
PREFIX_LM_ONNX ?= $(PI_ONNX_DIR)/prefix_lm.onnx
SUFFIX_STEP_ONNX ?= $(PI_ONNX_DIR)/suffix_step.onnx
PREFIX_EMBED_ENGINE ?= $(PI_ENGINE_DIR)/prefix_embed.engine
PREFIX_LM_ENGINE ?= $(PI_ENGINE_DIR)/prefix_lm.engine
SUFFIX_STEP_ENGINE ?= $(PI_ENGINE_DIR)/suffix_step.engine

FASTWAM_ONNX_DIR ?= assets/fastwam_libero/onnx
FASTWAM_ENGINE_DIR ?= assets/fastwam_libero/engines
FASTWAM_INT8_ONNX_DIR ?= assets/fastwam_libero_int8/onnx
FASTWAM_INT8_ENGINE_DIR ?= assets/fastwam_libero_int8/engines

SEMANTICVLA_ONNX_DIR ?= assets/semanticvla_libero/onnx
SEMANTICVLA_ENGINE_DIR ?= assets/semanticvla_libero/engines

EVO1_ONNX_DIR ?= assets/evo1_libero/onnx
EVO1_ENGINE_DIR ?= assets/evo1_libero/engines

SMOLVLA_ONNX_DIR ?= assets/smolvla_libero/onnx
SMOLVLA_ENGINE_DIR ?= assets/smolvla_libero/engines

DIT4DIT_ONNX_DIR ?= assets/dit4dit_libero/onnx
DIT4DIT_ENGINE_DIR ?= assets/dit4dit_libero/engines

GROOT_ONNX_DIR ?= assets/groot_n1d6_libero/onnx
GROOT_ENGINE_DIR ?= assets/groot_n1d6_libero/engines
STARVLA_ONNX_DIR ?= assets/starvla_libero/onnx
STARVLA_ENGINE_DIR ?= assets/starvla_libero/engines

VAE_IMAGE_ENCODER_ONNX ?= $(FASTWAM_ONNX_DIR)/vae_image_encoder.onnx
VIDEO_PREFILL_ONNX ?= $(FASTWAM_ONNX_DIR)/video_prefill.onnx
ACTION_STEP_ONNX ?= $(FASTWAM_ONNX_DIR)/action_step_dynamic_kv.onnx
VAE_IMAGE_ENCODER_ENGINE ?= $(FASTWAM_ENGINE_DIR)/vae_image_encoder.engine
VIDEO_PREFILL_ENGINE ?= $(FASTWAM_ENGINE_DIR)/video_prefill.engine
ACTION_STEP_ENGINE ?= $(FASTWAM_ENGINE_DIR)/action_step_dynamic_kv.engine

SEMANTICVLA_BACKBONE_ONNX ?= $(SEMANTICVLA_ONNX_DIR)/backbone.onnx
SEMANTICVLA_ACTION_STEP_ONNX ?= $(SEMANTICVLA_ONNX_DIR)/action_step.onnx
SEMANTICVLA_BACKBONE_ENGINE ?= $(SEMANTICVLA_ENGINE_DIR)/backbone.engine
SEMANTICVLA_ACTION_STEP_ENGINE ?= $(SEMANTICVLA_ENGINE_DIR)/action_step.engine

EVO1_BACKBONE_ONNX ?= $(EVO1_ONNX_DIR)/backbone.onnx
EVO1_ACTION_STEP_ONNX ?= $(EVO1_ONNX_DIR)/action_step.onnx
EVO1_BACKBONE_ENGINE ?= $(EVO1_ENGINE_DIR)/backbone.engine
EVO1_ACTION_STEP_ENGINE ?= $(EVO1_ENGINE_DIR)/action_step.engine

SMOLVLA_PREFIX_EMBED_ONNX ?= $(SMOLVLA_ONNX_DIR)/prefix_embed.onnx
SMOLVLA_PREFIX_LM_ONNX ?= $(SMOLVLA_ONNX_DIR)/prefix_lm.onnx
SMOLVLA_SUFFIX_LOOP_ONNX ?= $(SMOLVLA_ONNX_DIR)/suffix_loop.onnx
SMOLVLA_PREFIX_EMBED_ENGINE ?= $(SMOLVLA_ENGINE_DIR)/prefix_embed.engine
SMOLVLA_PREFIX_LM_ENGINE ?= $(SMOLVLA_ENGINE_DIR)/prefix_lm.engine
SMOLVLA_SUFFIX_LOOP_ENGINE ?= $(SMOLVLA_ENGINE_DIR)/suffix_loop.engine

DIT4DIT_VAE_ONNX ?= $(DIT4DIT_ONNX_DIR)/cosmos_vae_encode.onnx
DIT4DIT_FEATURE_ONNX ?= $(DIT4DIT_ONNX_DIR)/cosmos_feature.onnx
DIT4DIT_ACTION_LOOP_ONNX ?= $(DIT4DIT_ONNX_DIR)/action_loop.onnx
DIT4DIT_VAE_ENGINE ?= $(DIT4DIT_ENGINE_DIR)/cosmos_vae_encode.engine
DIT4DIT_FEATURE_ENGINE ?= $(DIT4DIT_ENGINE_DIR)/cosmos_feature.engine
DIT4DIT_ACTION_LOOP_ENGINE ?= $(DIT4DIT_ENGINE_DIR)/action_loop.engine

GROOT_BACKBONE_ONNX ?= $(GROOT_ONNX_DIR)/backbone.onnx
GROOT_ACTION_LOOP_ONNX ?= $(GROOT_ONNX_DIR)/action_loop.onnx
GROOT_BACKBONE_ENGINE ?= $(GROOT_ENGINE_DIR)/backbone.engine
GROOT_ACTION_LOOP_ENGINE ?= $(GROOT_ENGINE_DIR)/action_loop.engine
STARVLA_POLICY_ONNX ?= $(STARVLA_ONNX_DIR)/policy.onnx
STARVLA_POLICY_ENGINE ?= $(STARVLA_ENGINE_DIR)/policy.engine

.PHONY: configure engines_pi05 engines_pi05_int8 engines_fastwam engines_fastwam_int8 engines_semanticvla engines_evo1 engines_smolvla engines_dit4dit engines_groot engines_starvla clean help

configure:
	cmake -S . -B $(BUILD_DIR) -GNinja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

engines_pi05: $(PREFIX_EMBED_ENGINE) $(PREFIX_LM_ENGINE) $(SUFFIX_STEP_ENGINE)

engines_pi05_int8:
	$(MAKE) engines_pi05 PI_ONNX_DIR=$(PI05_INT8_ONNX_DIR) PI_ENGINE_DIR=$(PI05_INT8_ENGINE_DIR) PI05_PREFIX_EMBED_TRT_FLAGS="$(PI05_INT8_TRT_FLAGS)" PI05_TRT_FLAGS="$(PI05_INT8_TRT_FLAGS)"

engines_fastwam: $(VAE_IMAGE_ENCODER_ENGINE) $(VIDEO_PREFILL_ENGINE) $(ACTION_STEP_ENGINE)

engines_fastwam_int8:
	$(MAKE) engines_fastwam FASTWAM_ONNX_DIR=$(FASTWAM_INT8_ONNX_DIR) FASTWAM_ENGINE_DIR=$(FASTWAM_INT8_ENGINE_DIR)

engines_semanticvla: $(SEMANTICVLA_BACKBONE_ENGINE) $(SEMANTICVLA_ACTION_STEP_ENGINE)

engines_evo1: $(EVO1_BACKBONE_ENGINE) $(EVO1_ACTION_STEP_ENGINE)

engines_smolvla: $(SMOLVLA_PREFIX_EMBED_ENGINE) $(SMOLVLA_PREFIX_LM_ENGINE) $(SMOLVLA_SUFFIX_LOOP_ENGINE)

engines_dit4dit: $(DIT4DIT_VAE_ENGINE) $(DIT4DIT_FEATURE_ENGINE) $(DIT4DIT_ACTION_LOOP_ENGINE)

engines_groot: $(GROOT_BACKBONE_ENGINE) $(GROOT_ACTION_LOOP_ENGINE)

engines_starvla: $(STARVLA_POLICY_ENGINE)

$(PI_ENGINE_DIR) $(FASTWAM_ENGINE_DIR) $(SEMANTICVLA_ENGINE_DIR) $(EVO1_ENGINE_DIR) $(SMOLVLA_ENGINE_DIR) $(DIT4DIT_ENGINE_DIR) $(GROOT_ENGINE_DIR) $(STARVLA_ENGINE_DIR):
	mkdir -p $@

$(PREFIX_EMBED_ENGINE): | $(PI_ENGINE_DIR)
	trtexec --onnx=$(PREFIX_EMBED_ONNX) --saveEngine=$@ $(PI05_PREFIX_EMBED_TRT_FLAGS)

$(PREFIX_LM_ENGINE): | $(PI_ENGINE_DIR)
	trtexec --onnx=$(PREFIX_LM_ONNX) --saveEngine=$@ $(PI05_TRT_FLAGS)

$(SUFFIX_STEP_ENGINE): | $(PI_ENGINE_DIR)
	trtexec --onnx=$(SUFFIX_STEP_ONNX) --saveEngine=$@ $(PI05_TRT_FLAGS)

$(VAE_IMAGE_ENCODER_ENGINE): | $(FASTWAM_ENGINE_DIR)
	trtexec --onnx=$(VAE_IMAGE_ENCODER_ONNX) --saveEngine=$@ $(TRT_FLAGS)

$(VIDEO_PREFILL_ENGINE): | $(FASTWAM_ENGINE_DIR)
	trtexec --onnx=$(VIDEO_PREFILL_ONNX) --saveEngine=$@ $(TRT_FLAGS) --outputIOFormats=fp16:chw

$(ACTION_STEP_ENGINE): | $(FASTWAM_ENGINE_DIR)
	trtexec --onnx=$(ACTION_STEP_ONNX) --saveEngine=$@ $(TRT_FLAGS)

$(SEMANTICVLA_BACKBONE_ENGINE): | $(SEMANTICVLA_ENGINE_DIR)
	trtexec --onnx=$(SEMANTICVLA_BACKBONE_ONNX) --saveEngine=$@ --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference

$(SEMANTICVLA_ACTION_STEP_ENGINE): | $(SEMANTICVLA_ENGINE_DIR)
	trtexec --onnx=$(SEMANTICVLA_ACTION_STEP_ONNX) --saveEngine=$@ $(TRT_FLAGS)

$(EVO1_BACKBONE_ENGINE): | $(EVO1_ENGINE_DIR)
	trtexec --onnx=$(EVO1_BACKBONE_ONNX) --saveEngine=$@ $(EVO1_BACKBONE_TRT_FLAGS)

$(EVO1_ACTION_STEP_ENGINE): | $(EVO1_ENGINE_DIR)
	trtexec --onnx=$(EVO1_ACTION_STEP_ONNX) --saveEngine=$@ $(EVO1_ACTION_TRT_FLAGS)

$(SMOLVLA_PREFIX_EMBED_ENGINE): | $(SMOLVLA_ENGINE_DIR)
	trtexec --onnx=$(SMOLVLA_PREFIX_EMBED_ONNX) --saveEngine=$@ $(TRT_FLAGS)

$(SMOLVLA_PREFIX_LM_ENGINE): | $(SMOLVLA_ENGINE_DIR)
	trtexec --onnx=$(SMOLVLA_PREFIX_LM_ONNX) --saveEngine=$@ --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference

$(SMOLVLA_SUFFIX_LOOP_ENGINE): | $(SMOLVLA_ENGINE_DIR)
	trtexec --onnx=$(SMOLVLA_SUFFIX_LOOP_ONNX) --saveEngine=$@ --builderOptimizationLevel=3 --memPoolSize=workspace:4096 --skipInference

$(DIT4DIT_VAE_ENGINE): | $(DIT4DIT_ENGINE_DIR)
	trtexec --onnx=$(DIT4DIT_VAE_ONNX) --saveEngine=$@ $(TRT_FLAGS)

$(DIT4DIT_FEATURE_ENGINE): | $(DIT4DIT_ENGINE_DIR)
	trtexec --onnx=$(DIT4DIT_FEATURE_ONNX) --saveEngine=$@ $(DIT4DIT_FEATURE_TRT_FLAGS)

$(DIT4DIT_ACTION_LOOP_ENGINE): | $(DIT4DIT_ENGINE_DIR)
	trtexec --onnx=$(DIT4DIT_ACTION_LOOP_ONNX) --saveEngine=$@ $(DIT4DIT_ACTION_TRT_FLAGS)

$(GROOT_BACKBONE_ENGINE): | $(GROOT_ENGINE_DIR)
	trtexec --onnx=$(GROOT_BACKBONE_ONNX) --saveEngine=$@ --stronglyTyped --builderOptimizationLevel=3 --memPoolSize=workspace:8192 --skipInference

$(GROOT_ACTION_LOOP_ENGINE): | $(GROOT_ENGINE_DIR)
	trtexec --onnx=$(GROOT_ACTION_LOOP_ONNX) --saveEngine=$@ $(GROOT_ACTION_TRT_FLAGS)

$(STARVLA_POLICY_ENGINE): | $(STARVLA_ENGINE_DIR)
	trtexec --onnx=$(STARVLA_POLICY_ONNX) --saveEngine=$@ $(STARVLA_POLICY_TRT_FLAGS)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  make engines_pi05     Convert PI0.5 ONNX files into TensorRT engines"
	@echo "  make engines_pi05_int8 Convert PI0.5-INT8 ONNX files into TensorRT engines"
	@echo "  make engines_fastwam  Convert FastWAM ONNX files into TensorRT engines"
	@echo "  make engines_fastwam_int8 Convert FastWAM-INT8 ONNX files into TensorRT engines"
	@echo "  make engines_semanticvla  Convert SemanticVLA ONNX files into TensorRT engines"
	@echo "  make engines_evo1     Convert Evo-1 ONNX files into TensorRT engines"
	@echo "  make engines_smolvla  Convert SmolVLA ONNX files into TensorRT engines"
	@echo "  make engines_dit4dit  Convert DiT4DiT ONNX files into TensorRT engines"
	@echo "  make engines_groot    Convert GR00T ONNX files into TensorRT engines"
	@echo "  make engines_starvla  Convert StarVLA ONNX files into TensorRT engines"
	@echo "  make clean        Remove build directory"
	@echo ""
	@echo "Config variables:"
	@echo "  PI_ONNX_DIR=$(PI_ONNX_DIR)"
	@echo "  PI_ENGINE_DIR=$(PI_ENGINE_DIR)"
	@echo "  PI05_INT8_ONNX_DIR=$(PI05_INT8_ONNX_DIR)"
	@echo "  PI05_INT8_ENGINE_DIR=$(PI05_INT8_ENGINE_DIR)"
	@echo "  FASTWAM_ONNX_DIR=$(FASTWAM_ONNX_DIR)"
	@echo "  FASTWAM_ENGINE_DIR=$(FASTWAM_ENGINE_DIR)"
	@echo "  FASTWAM_INT8_ONNX_DIR=$(FASTWAM_INT8_ONNX_DIR)"
	@echo "  FASTWAM_INT8_ENGINE_DIR=$(FASTWAM_INT8_ENGINE_DIR)"
	@echo "  SEMANTICVLA_ONNX_DIR=$(SEMANTICVLA_ONNX_DIR)"
	@echo "  SEMANTICVLA_ENGINE_DIR=$(SEMANTICVLA_ENGINE_DIR)"
	@echo "  EVO1_ONNX_DIR=$(EVO1_ONNX_DIR)"
	@echo "  EVO1_ENGINE_DIR=$(EVO1_ENGINE_DIR)"
	@echo "  SMOLVLA_ONNX_DIR=$(SMOLVLA_ONNX_DIR)"
	@echo "  SMOLVLA_ENGINE_DIR=$(SMOLVLA_ENGINE_DIR)"
	@echo "  DIT4DIT_ONNX_DIR=$(DIT4DIT_ONNX_DIR)"
	@echo "  DIT4DIT_ENGINE_DIR=$(DIT4DIT_ENGINE_DIR)"
	@echo "  GROOT_ONNX_DIR=$(GROOT_ONNX_DIR)"
	@echo "  GROOT_ENGINE_DIR=$(GROOT_ENGINE_DIR)"
	@echo "  STARVLA_ONNX_DIR=$(STARVLA_ONNX_DIR)"
	@echo "  STARVLA_ENGINE_DIR=$(STARVLA_ENGINE_DIR)"
	@echo "  TRTEXEC=trtexec"
	@echo "  TRT_FLAGS=$(TRT_FLAGS)"
	@echo "  PI05_PREFIX_EMBED_TRT_FLAGS=$(PI05_PREFIX_EMBED_TRT_FLAGS)"
	@echo "  PI05_TRT_FLAGS=$(PI05_TRT_FLAGS)"
	@echo "  STARVLA_POLICY_TRT_FLAGS=$(STARVLA_POLICY_TRT_FLAGS)"
