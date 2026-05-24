"""
Thin entrypoint for finetuning runs.
Loads centralized config, initializes pipeline state and run context,
and starts the FinetuneTrainer.
"""
from configs.loader import load_config
from modules.utils.run_context import RunContext
from modules.utils.pipeline_state import PipelineState
from finetune.core.finetune_trainer import FinetuneTrainer

def main():
    # Load finetune config from centralized loader
    config = load_config("finetune")
    # Create run context (timestamps, run id, environment)
    run_ctx = RunContext.from_config(config)
    # PipelineState tracks stage/substage offsets and current checkpoint id
    state = PipelineState(run_ctx.run_id, stage="finetune")

    trainer = FinetuneTrainer(config, state, run_ctx)
    trainer.run()

if __name__ == "__main__":
    main()
