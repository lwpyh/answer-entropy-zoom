import os
import json
import hydra
from omegaconf import OmegaConf
from pprint import pprint

@hydra.main(config_path="../trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # ---- tokenizer / processor ----
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    # ---- build dataloader directly ----
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    dummy = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping={},
        resource_pool_manager=None,
        ray_worker_group_cls=None,
        reward_fn=None,
        val_reward_fn=None,
    )

    dummy._create_dataloader()
    val_loader = dummy.val_dataloader

    # ---- build vLLM engine directly ----
    from verl.workers.vllm_rollout import VLLMRollout

    rollout = VLLMRollout(config.actor_rollout_ref.rollout, tokenizer, processor)

    save_dir = config.trainer.validation_data_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "generations.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for batch in val_loader:
            outputs = rollout.generate_sequences(batch)

            if hasattr(outputs, "to_dict"):
                record = outputs.to_dict()
            else:
                record = {"raw": str(outputs)}

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved to {save_path}")

if __name__ == "__main__":
    main()
