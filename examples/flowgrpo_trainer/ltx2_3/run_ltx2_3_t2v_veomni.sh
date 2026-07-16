# LTX-2.3 full-weight RL with the VeOmni engine, vllm_omni rollout.
#
# This recipe mirrors run_qwen_image_ocr_veomni.sh but adapts for LTX-2.3's
# multimodal video/audio generation.  Key differences from Qwen-Image:
#
#   - architecture=LTXVideoTransformerModel (no model_index.json; set explicitly)
#   - transformer_subfolder="" (LTX-2.3 weights are at the model root)
#   - pipeline.num_frames=41 (video generation, not single image)
#   - pipeline.height/width set for video resolution
#   - external_lib set to import the LTX-2.3 pipeline adapters
#
# Requires VeOmni installed alongside the verl-omni base environment; see
# docs/start/install.md "Optional engine backends" for the install workaround.
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

# The dataset should contain pre-computed video latents and prompt embeddings,
# as LTX-2.3 requires offline pre-processing (see VeOmni's precompute scripts).
train_path=$WORKSPACE/data/ltx2_3/train.parquet
test_path=$WORKSPACE/data/ltx2_3/test.parquet

model_name=LTX-2.3

reward_model_name=Qwen/Qwen3-VL-8B-Instruct
export custom_reward_model_path=$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors
custom_reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
ACTOR_SP=1
ROLLOUT_TP=1
REWARD_TP=4

# Video generation parameters
VIDEO_HEIGHT=512
VIDEO_WIDTH=512
VIDEO_NUM_FRAMES=41
VIDEO_FPS=24
NUM_INFERENCE_STEPS=30

ENGINE=vllm_omni
REWARD_ENGINE=vllm
TRAINER_BACKEND=veomni


python3 -m verl_omni.trainer.main_diffusion \
    diffusion/model_engine=veomni_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.architecture=LTXVideoTransformerModel \
    actor_rollout_ref.model.transformer_subfolder="" \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.ltx2_3_flow_grpo \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.actor.veomni_config.ulysses_parallel_size=$ACTOR_SP \
    actor_rollout_ref.actor.veomni_config.param_offload=True \
    actor_rollout_ref.actor.veomni_config.optimizer_offload=True \
    actor_rollout_ref.actor.veomni_config.attn_implementation=eager \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$VIDEO_HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$VIDEO_WIDTH \
    actor_rollout_ref.rollout.pipeline.num_frames=$VIDEO_NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$NUM_INFERENCE_STEPS \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.veomni_config.strategy=$TRAINER_BACKEND \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_video \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=ltx2_3_video_${TRAINER_BACKEND} \
    trainer.log_val_generations=4 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / NUM_NODES)) \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
