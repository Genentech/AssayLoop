"""wandb sweep agent entrypoint for from-scratch RL ranker training.

A wandb sweep agent runs this module once per trial. ``wandb.init()`` picks up
the trial's sampled hyperparameters (defined in ``configs/sweeps/ranker_rl_*.yaml``)
into ``wandb.config``; we map those onto :func:`run_rl_training` kwargs and let it
attach to the *same* run (``reuse_wandb=True``) so all train/eval curves and the
summary objective land on the trial.

Run it with::

    wandb sweep configs/sweeps/ranker_rl_scratch.yaml      # -> prints SWEEP_ID
    # one agent per GPU (each trial uses a single GPU):
    CUDA_VISIBLE_DEVICES=0 wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID> &
    CUDA_VISIBLE_DEVICES=1 wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID> &
    ...

Only single-GPU trials are supported here (no torchrun/DDP); parallelism comes
from running multiple agents, which is the natural fit for a hyperparameter sweep.
"""

from __future__ import annotations

import logging

import wandb

from assayloop.amortized.rl import run_rl_training

log = logging.getLogger("assayloop.sweep_ranker_rl")

# Hyperparameters the sweep is allowed to set. Anything in wandb.config that is
# also a run_rl_training kwarg and listed here is forwarded; everything else uses
# the fixed defaults below. (Keeps a typo in the YAML from silently doing nothing
# *and* from being forwarded as an unexpected kwarg.)
SWEEPABLE = {
    # architecture (from-scratch transformer)
    "d_model", "d_gene", "d_hit", "nhead", "num_layers", "dim_feedforward", "dropout",
    "encoder_type", "disable_gene_bias", "use_description",
    # RL objective + optimization
    "reward_mode", "ctx_reset_epochs", "aux_hit_coef", "kl_coef", "ent_coef",
    "terminal_coef", "adv_std_floor", "adv_clip", "lr", "weight_decay",
    "group_size", "n_steps", "batch_size", "gamma", "temperature", "grad_clip",
    "rl_max_context", "epochs", "seed",
}

# Fixed defaults for a from-scratch transformer RL sweep. Sized for a quick-ish
# trial that still runs past the ~e100 region where the unanchored collapse
# appeared, so stability shows up in the eval curve / max_eval_drop.
BASE = dict(
    from_scratch=True,
    encoder_type="transformer",
    disable_gene_bias=True,
    use_description=False,
    train_screen_set="public_train",
    eval_screen_set="public_validation",
    eval_screens=0,        # 0 = score the entire public_validation split (only ~20 screens)
    train_eval_screens=30,
    eval_every=1,
    epochs=150,
    group_size=8,
    n_steps=10,
    batch_size=100,
    rl_max_context=1024,  # feed the full 10x100 context (run_rl_training default is 512)
    lr=3e-4,
    reward_mode="telescope",
    aux_hit_coef=0.0,
    kl_coef=0.0,
    ent_coef=0.01,
    adv_std_floor=1e-6,
    adv_clip=0.0,
    device="cuda",
    wandb_mode="online",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run = wandb.init()
    cfg = dict(run.config)

    kwargs = dict(BASE)
    for k, v in cfg.items():
        if k in SWEEPABLE:
            kwargs[k] = v

    # from-scratch RL forces kl_coef=0 internally; surface a hint if the YAML set it.
    if kwargs.get("from_scratch") and kwargs.get("kl_coef", 0.0):
        log.info("from_scratch trial: kl_coef=%s will be forced to 0 by run_rl_training.",
                 kwargs["kl_coef"])

    run_name = run.name or f"sweep-rl-{run.id}"
    log.info("Sweep trial %s -> run_rl_training kwargs: %s", run.id,
             {k: kwargs[k] for k in sorted(SWEEPABLE) if k in kwargs})

    run_rl_training(run_name=run_name, reuse_wandb=True, **kwargs)


if __name__ == "__main__":
    main()
