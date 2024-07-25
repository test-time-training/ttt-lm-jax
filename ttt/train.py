import mlxu
import wandb
import os.path as osp

from tqdm import tqdm
from copy import deepcopy

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.experimental.multihost_utils import process_allgather
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict

from ttt.infra.optimizers import OptimizerFactory
from ttt.dataloader.language_modeling_hf import LMDataModule
from ttt.infra.checkpoint import StreamingCheckpointer
from ttt.models.model import ModelConfig, CausalLM
from ttt.infra.jax_utils import (
    JaxRNG,
    JaxDistributedConfig,
    next_rng,
    match_partition_rules,
    cross_entropy_loss_and_accuracy,
    global_norm,
    get_float_dtype_by_name,
    set_random_seed,
    average_metrics,
    get_weight_decay_mask,
    make_shard_and_gather_fns,
    with_sharding_constraint,
    master_print,
    log_ttt_stats,
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=0,
    mesh_dim="-1,64,1",
    dtype="fp32",
    eval_mode=False,
    load_part="trainstate",
    total_steps=100,
    load_model_config="",
    update_model_config="",
    save_checkpoint_freq=100,
    save_milestone_freq=0,
    dataset_path="",
    dataset_name="the_pile",
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    seq_length=2048,
    global_batch_size=1,
    accum_steps=1,
    loader_workers=48,
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    exp_dir="",
    exp_name="",
    resume_exp_name="",
    resume_step="",
    jax_distributed=JaxDistributedConfig.get_default_config(),
    is_rollback_reshuffle=False,
)


def make_train_step_fn(model, optimizer_info, model_config, accum_steps=1):

    if accum_steps == 1:

        def train_step(train_state, rng, batch, ttt_lr_mult, output_ttt_stats=False):
            rng_generator = JaxRNG(rng)
            batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))

            def loss_and_accuracy(params):
                outputs = model.apply(
                    params,
                    batch["input_tokens"],
                    ttt_lr_mult=ttt_lr_mult,
                    deterministic=False,
                    output_ttt_stats=output_ttt_stats,
                    rngs=rng_generator(model_config.rng_keys()),
                )
                logits = outputs.logits
                ttt_stats = outputs.ttt_stats
                loss, _ = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
                return loss, ttt_stats

            grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
            (loss, ttt_stats), grads = grad_fn(train_state.params)

            train_state = train_state.apply_gradients(grads=grads)
            learning_rate = optimizer_info["learning_rate_schedule"](train_state.step)
            grads_norm = global_norm(grads)

            return (train_state, loss, ttt_stats, grads_norm, learning_rate, rng_generator())

    elif accum_steps > 1:

        def train_step(train_state, rng, batch, ttt_lr_mult, output_ttt_stats=False):
            rng_generator = JaxRNG(rng)
            rngs = rng_generator(model_config.rng_keys())

            def computation(carry, micro_batch):
                sum_grads = carry["sum_grads"]
                micro_batch = with_sharding_constraint(micro_batch, PS(("dp", "fsdp")))

                def loss_and_accuracy(params):
                    outputs = model.apply(
                        params,
                        micro_batch["input_tokens"],
                        ttt_lr_mult=ttt_lr_mult,
                        deterministic=False,
                        output_ttt_stats=output_ttt_stats,
                        rngs=rngs,
                    )
                    logits = outputs.logits
                    ttt_stats = outputs.ttt_stats
                    loss, _ = cross_entropy_loss_and_accuracy(
                        logits, micro_batch["target_tokens"], micro_batch["loss_masks"]
                    )
                    return loss, ttt_stats

                grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
                (loss, ttt_stats), grads = grad_fn(train_state.params)
                sum_grads = tree_map(lambda x, y: x + y, sum_grads, grads)
                carry_new = {"sum_grads": sum_grads}
                return carry_new, (loss, ttt_stats)

            sum_grads = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), train_state.params)
            carry_init = {"sum_grads": sum_grads}
            batch = tree_map(lambda x: x.reshape(FLAGS.accum_steps, -1, *x.shape[1:]), batch)
            carry_new, outputs = jax.lax.scan(computation, carry_init, batch)
            loss, ttt_stats = outputs
            loss = jnp.mean(loss)
            if output_ttt_stats:
                ttt_stats = tree_map(lambda x: jnp.mean(x, axis=0), ttt_stats)
            else:
                ttt_stats = None
            grads = jax.tree_util.tree_map(lambda x: x / FLAGS.accum_steps, carry_new["sum_grads"])

            train_state = train_state.apply_gradients(grads=grads)
            learning_rate = optimizer_info["learning_rate_schedule"](train_state.step)
            grads_norm = global_norm(grads)

            return (train_state, loss, ttt_stats, grads_norm, learning_rate, rng_generator())

    else:
        raise ValueError(f"Accum steps must >= 1, got {accum_steps}")

    return train_step


def make_eval_step_fn(model, model_config):
    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(("dp", "fsdp")))
        logits = model.apply(
            train_state.params, batch["input_tokens"], deterministic=True, rngs=rng_generator(model_config.rng_keys())
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(logits, batch["target_tokens"], batch["loss_masks"])
        metrics = dict(eval_loss=loss, eval_accuracy=accuracy)
        return rng_generator(), metrics

    return eval_step


def make_sharded_functions(model, optimizer, optimizer_info, model_config):
    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, FLAGS.seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, FLAGS.seq_length), dtype=jnp.int32),
            rngs=rng_generator(model_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    train_step = make_train_step_fn(model, optimizer_info, model_config, FLAGS.accum_steps)

    train_state_shapes = jax.eval_shape(init_fn, next_rng())

    train_state_partition = match_partition_rules(model_config.get_partition_rules(), train_state_shapes)

    shard_fns, gather_fns = make_shard_and_gather_fns(train_state_partition, train_state_shapes)

    sharded_init_fn = pjit(init_fn, in_shardings=PS(), out_shardings=train_state_partition)

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params,),
        out_shardings=train_state_partition,
        donate_argnums=(0,),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS(), PS(), PS(), PS()),
        static_argnums=4,
        donate_argnums=(0,),
    )

    return (
        sharded_init_fn,
        sharded_create_trainstate_from_params,
        sharded_train_step,
        shard_fns,
        gather_fns,
        train_state_shapes,
        train_state_partition,
    )


def make_save_checkpoint(checkpointer, gather_fns, variant, flags_config_dict, model_config, global_batch_size):
    def save_checkpoint(train_state, train_loader, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(step=step, variant=variant, flags=flags_config_dict, model_config=model_config.to_dict())
        sampler_state_dict = {
            "random_state": train_loader.sampler.state_dict()["random_state"],
            "shuffle_log": train_loader.sampler.state_dict()["shuffle_log"],
            "counter": step * global_batch_size,
        }
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=deepcopy(sampler_state_dict),
            milestone=milestone,
        )

    return save_checkpoint


def make_get_ttt_lr_mult(model_config):

    if (
        hasattr(model_config, "ttt_base_lr_init")
        and model_config.ttt_base_lr_init > 0
        and model_config.ttt_base_lr_warmup > 0
    ):
        ttt_lr_mult_warmup_steps = model_config.ttt_base_lr_warmup
        ttt_lr_mult_init = model_config.ttt_base_lr_init
        ttt_lr_mult_peak = model_config.ttt_base_lr

        def get_ttt_lr_mult(step):
            ttt_lr_mult = ttt_lr_mult_init + min(1.0, (step - 1) / ttt_lr_mult_warmup_steps) * (
                ttt_lr_mult_peak - ttt_lr_mult_init
            )
            ttt_lr_mult = ttt_lr_mult / ttt_lr_mult_peak * jnp.ones((1,), dtype=jnp.bfloat16)
            return ttt_lr_mult

    else:

        def get_ttt_lr_mult(step):
            ttt_lr_mult = jnp.ones((1,), dtype=jnp.bfloat16)
            return ttt_lr_mult

    return get_ttt_lr_mult


def initialize_or_resume(
    checkpointer,
    train_loader,
    train_state_shapes,
    sharded_init_fn,
    shard_fns,
    sharded_create_trainstate_from_params,
    FLAGS,
):
    start_step = 1
    train_state, restored_params = None, None
    if FLAGS.resume_exp_name != "":
        assert FLAGS.load_part in ["trainstate", "trainstate_params"]
        ckpt_resume_dir = (
            FLAGS.load_part
            + "::"
            + osp.join(
                FLAGS.exp_dir,
                FLAGS.resume_exp_name,
                (
                    f"step_{int(FLAGS.resume_step)}/streaming_train_state_{int(FLAGS.resume_step)}"
                    if FLAGS.resume_step
                    else "streaming_train_state"
                ),
            )
        )
        train_state, restored_params = checkpointer.load_trainstate_checkpoint(
            ckpt_resume_dir, train_state_shapes, shard_fns
        )

        if FLAGS.load_part == "trainstate":
            start_step = int(jax.device_get(train_state.step)) + 1
            master_print(f"Resuming training from checkpoint at step {start_step - 1}...")
            dataset_pkl_filename = (
                f"step_{int(FLAGS.resume_step)}/dataset_{int(FLAGS.resume_step)}.pkl"
                if FLAGS.resume_step
                else "dataset.pkl"
            )
            dataset_resume_dir = osp.join(FLAGS.exp_dir, FLAGS.resume_exp_name, dataset_pkl_filename)
            train_loader.sampler.load_state_dict(deepcopy(mlxu.load_pickle(dataset_resume_dir)))

        if FLAGS.is_rollback_reshuffle:
            train_loader.sampler.is_rollback = True

    if train_state is None and restored_params is None:
        train_state = sharded_init_fn(next_rng())
    elif train_state is None and restored_params is not None:
        train_state = sharded_create_trainstate_from_params(restored_params)
        del restored_params

    return start_step, train_state, train_loader


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    set_random_seed(FLAGS.seed)
    process_num = jax.process_count()
    global_dev_num = jax.device_count()
    local_dev_num = jax.local_device_count()
    master_process = jax.process_index() == 0

    dev_info = f"Process # {process_num}\tLocal dev # {local_dev_num}\tTotal dev # {global_dev_num}"
    master_print(dev_info)

    seq_length = FLAGS.seq_length
    global_batch_size = FLAGS.global_batch_size
    is_rollback_reshuffle = FLAGS.is_rollback_reshuffle

    # Create dataloader
    data_module = LMDataModule(
        dataset_name=FLAGS.dataset_name,
        dataset_config_name=None,
        tokenizer_name=FLAGS.tokenizer_name,
        cache_dir=FLAGS.dataset_path,
        max_length=seq_length,
        add_eos=True,
        batch_size=global_batch_size,
        batch_size_eval=global_batch_size,
        loader_workers=FLAGS.loader_workers,
        shuffle=True,
        fault_tolerant=True,
        drop_last=True,
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Update model model_config
    if FLAGS.load_model_config != "":
        model_config = ModelConfig.load_config(FLAGS.load_model_config)
    else:
        raise RuntimeError(f"model_config must be specified")
    if FLAGS.update_model_config:
        update_dic = eval(FLAGS.update_model_config)
        for key, value in update_dic.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            else:
                raise KeyError(f"Update key {key} not in model_config")
    model_config.vocab_size = data_module.vocab_size
    model_config.max_sequence_length = seq_length
    flags_config_dict.model_config = model_config

    # Create WandB run and checkpointer
    if master_process:
        wandb.init(project="TTT-LM", config=flags_config_dict, name=FLAGS.exp_name)
    ckpt_dir = osp.join(FLAGS.exp_dir, FLAGS.exp_name)
    checkpointer = StreamingCheckpointer(FLAGS.checkpointer, ckpt_dir, enable=master_process)

    # Create model and optimizer
    model = CausalLM(model_config, dtype=get_float_dtype_by_name(FLAGS.dtype))
    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer, get_weight_decay_mask(model_config.get_weight_decay_exclusions())
    )

    # Helper function for dynamic TTT learning rate
    get_ttt_lr_mult = make_get_ttt_lr_mult(model_config)

    # Create sharded train functions
    (
        sharded_init_fn,
        sharded_create_trainstate_from_params,
        sharded_train_step,
        shard_fns,
        gather_fns,
        train_state_shapes,
        train_state_partition,
    ) = make_sharded_functions(model, optimizer, optimizer_info, model_config)

    save_checkpoint = make_save_checkpoint(
        checkpointer, gather_fns, variant, flags_config_dict, model_config, global_batch_size
    )

    mesh = model_config.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        sharded_rng = next_rng()

        start_step, train_state, train_loader = initialize_or_resume(
            checkpointer,
            train_loader,
            train_state_shapes,
            sharded_init_fn,
            shard_fns,
            sharded_create_trainstate_from_params,
            FLAGS,
        )

        if FLAGS.eval_mode:
            eval_step = make_eval_step_fn(model, model_config)
            sharded_eval_step = pjit(
                eval_step,
                in_shardings=(train_state_partition, PS(), PS()),
                out_shardings=(PS(), PS()),
                donate_argnums=(1,),
            )

            val_loader = data_module.val_dataloader()
            eval_metric_list = []

            for eval_batch in tqdm(val_loader, disable=not master_process):
                for k in eval_batch.keys():
                    eval_batch[k] = eval_batch[k].numpy()
                sharded_rng, eval_metrics = sharded_eval_step(train_state, sharded_rng, eval_batch)
                eval_metric_list.append(eval_metrics)

            val_loss_avg = average_metrics(process_allgather(eval_metric_list))["eval_loss"].item()
            master_print(f"Eval Loss: {val_loss_avg:.4f}")
            exit(0)

        train_loader_iterator = iter(train_loader)

        for step in tqdm(
            range(start_step, FLAGS.total_steps + 1),
            initial=start_step,
            total=FLAGS.total_steps,
            disable=not master_process,
            desc=f"Training {FLAGS.exp_name}",
        ):
            try:
                batch = next(train_loader_iterator)
            except StopIteration:
                train_loader.sampler.counter = 0
                train_loader_iterator = iter(train_loader)
                batch = next(train_loader_iterator)

            if is_rollback_reshuffle:
                sampler_state_dict = {
                    "random_state": train_loader.sampler.state_dict()["random_state"],
                    "shuffle_log": train_loader.sampler.state_dict()["shuffle_log"],
                    "counter": (step - 1) * global_batch_size,
                }
                if master_process and FLAGS.resume_exp_name != "":
                    master_print("Updating sampler state after rollback...")
                    dataset_pkl_filename = (
                        f"step_{int(FLAGS.resume_step)}/dataset_{int(FLAGS.resume_step)}.pkl"
                        if FLAGS.resume_step
                        else "dataset_state.pkl"
                    )
                    dataset_resume_dir = osp.join(FLAGS.exp_dir, FLAGS.resume_exp_name, dataset_pkl_filename)
                    mlxu.save_pickle(deepcopy(sampler_state_dict), dataset_resume_dir)
                is_rollback_reshuffle = False
                master_print("Finished updating sampler state.")

            for k in batch.keys():
                batch[k] = batch[k].numpy()

            ttt_lr_mult = get_ttt_lr_mult(step)
            output_ttt_stats = (
                FLAGS.save_milestone_freq > 0
                and step % FLAGS.save_milestone_freq == 0
                and model_config.seq_modeling_block != "self_attention"
            )

            train_state, loss, ttt_stats, grads_norm, learning_rate, sharded_rng = sharded_train_step(
                train_state, sharded_rng, batch, ttt_lr_mult, output_ttt_stats
            )

            if master_process:
                wandb.log(
                    {
                        "Train Loss": loss.item(),
                        "Gradient Norm": grads_norm.item(),
                        "Learning Rate": learning_rate.item(),
                    },
                    step=step,
                )

                if output_ttt_stats:
                    for layer in range(len(ttt_stats)):
                        ttt_stats_layer = process_allgather(ttt_stats[layer])
                        n_mini_batch = len(ttt_stats_layer[0])
                        x_axis = [model_config.mini_batch_size * i for i in range(1, n_mini_batch + 1)]
                        log_ttt_stats(layer, ttt_stats_layer, x_axis, step)

            if (FLAGS.save_checkpoint_freq > 0 and step % FLAGS.save_checkpoint_freq == 0) or (
                step == FLAGS.total_steps
            ):
                master_print(f"Saving checkpoint at step {step}, do not kill...")
                save_checkpoint(train_state, train_loader, step % FLAGS.save_milestone_freq == 0)

            if step == FLAGS.total_steps:
                master_print("Training has completed!")


if __name__ == "__main__":
    mlxu.run(main)
