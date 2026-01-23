# ------------------------------------------------------------
#  Experiment Metadata
# ------------------------------------------------------------
PROJECT_DIR="$(pwd)"
PROJECT_NAME="CodeVision"
EXP_NAME="qwen3vl8b"
SAVE_DIR="./saves/${PROJECT_NAME}/${EXP_NAME}"
RUNTIME_ENV="verl/trainer/runtime_env.yaml"

# ------------------------------------------------------------
#  Ray & Cluster Settings
# ------------------------------------------------------------
export RAY_ADDRESS="RAY_ADDRESS_HERE"
export LLM_JUDGE_BASE_URL="YOUR_LLM_JUDGE_BASE_URL"
NNODES=2
NGPUS_PER_NODE=8

# ------------------------------------------------------------
#  Model & Data Paths
# ------------------------------------------------------------
# The model you want to evaluate
MODEL_PATH="Qwen/Qwen3-VL-8B-Thinking"

# https://huggingface.co/datasets/kkwok/CodeVision-RL
# train_files="['train_group_1.parquet','train_group_2.parquet','train_group_3.parquet','train_group_4.parquet']"

# Put all your benchmark here for evaluation
test_files="['mvtoolbench.parquet']"

# ------------------------------------------------------------
#  Core Algorithm Hyper-parameters
# ------------------------------------------------------------
# Algorithm
adv_estimator="grpo"
loss_agg_mode="token-mean"
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Config File
cfg_path="${PROJECT_DIR}/recipe/codevision/config"
cfg_name="grpo_trainer"
tool_cfg_path="recipe/codevision/config/code_image_tool_config.yaml"
new_sp_path="recipe/codevision/config/sp.txt"

# Batch Size
train_bsz=64
train_mini_bsz=32

train_micro_bsz_per_gpu=1
infer_micro_bsz_per_gpu=1
n_resp_per_prompt=8
max_turns=12

# Sequence Length
max_prompt_len=$((1024 * 16))
max_resp_len=$((1024 * 16))
max_tool_resp_len=$((1024 * 10))
max_image_resolution=$((1024 * 8 * 28 * 28))

# Performance Related Parameter
offload=True
train_sp_size=1
infer_tp_size=4
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_len + max_resp_len))
infer_ppo_max_token_len=$((max_prompt_len + max_resp_len))
max_num_batched_tokens=$((max_prompt_len + max_resp_len))

# Reward Model
tool_reward_enable=True
tool_reward_alpha=1.0
tool_reward_beta=0.0
tool_reward_gamma=0.5
tool_reward_delta=0.5
format_reward_weight=0.1
exec_reward_weight=0.0
emerge_reward_weight=0.2

exploration_reward_enable=False
exploration_reward_weight=0.0
exploration_decay_steps=0

# Trainer Schedule & Logging
val_before_train=True
test_freq=20
save_freq=400
total_epochs=1
log_val_generations=8
log_train_generations=8
log_train_freq=20


# Only test mode, training will not start
ONLY_TEST=True
VAL_METRICS_OUTPUT="./saves/CodeVision/qwen3vl8b/metrics.json"
val_bsz=256



ray job submit --no-wait \
    --working-dir $(pwd) \
    --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path=${cfg_path} \
    --config-name=${cfg_name} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_batch_size=${train_bsz} \
    data.val_batch_size=${val_bsz} \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_resp_len} \
    data.filter_overlong_prompts=False \
    data.truncation="error" \
    data.return_raw_chat=True \
    data.train_files=${train_files} \
    data.val_files=${test_files}  \
    data.return_multi_modal_inputs=False \
    +data.replace_system_prompt=True \
    +data.new_sp_path=${new_sp_path} \
    +data.enable_image_resize=True \
    +data.max_image_resolution=${max_image_resolution} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${train_sp_size} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp_size} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_resp_len} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_cfg_path} \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    +reward_model.tool_reward.enable=${tool_reward_enable} \
    +reward_model.tool_reward.alpha=${tool_reward_alpha} \
    +reward_model.tool_reward.beta=${tool_reward_beta} \
    +reward_model.tool_reward.gamma=${tool_reward_gamma} \
    +reward_model.tool_reward.delta=${tool_reward_delta} \
    +reward_model.format_reward_weight=${format_reward_weight} \
    +reward_model.exec_reward_weight=${exec_reward_weight} \
    +reward_model.emerge_reward_weight=${emerge_reward_weight} \
    +reward_model.exploration_reward.enable=${exploration_reward_enable} \
    +reward_model.exploration_reward.weight=${exploration_reward_weight} \
    +reward_model.exploration_reward.decay_steps=${exploration_decay_steps} \
    trainer.critic_warmup=0 \
    trainer.val_before_train=${val_before_train} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    +trainer.log_train_freq=${log_train_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.resume_mode=auto \
    +trainer.only_test=${ONLY_TEST} \
    +trainer.val_metrics_output=${VAL_METRICS_OUTPUT} 