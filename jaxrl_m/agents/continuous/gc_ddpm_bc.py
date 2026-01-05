import copy
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import equinox as eqx

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper, LCEncodingWrapper, Imaginations_LCEncodingWrapper, Imaginations_LCEncodingWrapper_Attention, Imaginations_LCEncodingWrapper_MultiLayerMultiHeadAttention, DiTEncoder

from jaxrl_m.networks.diffusion_nets import (
    FourierFeatures,
    cosine_beta_schedule,
    vp_beta_schedule,
    ScoreActor,
    MMDiTScoreActor,
    InContextMMDiTScoreActor
)
from jaxrl_m.networks.mlp import MLP, MLPResNet
from mmdit_jax import MMDiT, InContextMMDiT


def ddpm_bc_loss(noise_prediction, noise):
    ddpm_loss = jnp.square(noise_prediction - noise).sum(-1)

    return (
        ddpm_loss.mean(),
        {"ddpm_loss": ddpm_loss, "ddpm_loss_mean": ddpm_loss.mean()},
    )


class GCDDPMBCAgent(flax.struct.PyTreeNode):
    """
    Models action distribution with a diffusion model.

    Assumes observation histories as input and action sequences as output.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def actor_loss_fn(params, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            noise_pred = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (batch["observations"], batch["goals"]),
                noisy_actions,
                time,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )

            return ddpm_bc_loss(noise_pred, noise_sample)

        loss_fns = {"actor": actor_loss_fn}

        # Store old params for change tracking
        old_params = self.state.params
        
        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Log parameter changes and gradient norms
        new_params = new_state.params
        
        # Track encoder parameters (handle both 'actor' and 'modules_actor' keys)
        actor_key = 'modules_actor' if 'modules_actor' in old_params else 'actor'
        if actor_key in old_params and 'encoder' in old_params[actor_key]:
            encoder_old = jax.tree_util.tree_leaves(old_params[actor_key]['encoder'])
            encoder_new = jax.tree_util.tree_leaves(new_params[actor_key]['encoder'])
            encoder_old_arrays = [x for x in encoder_old if isinstance(x, jnp.ndarray)]
            encoder_new_arrays = [x for x in encoder_new if isinstance(x, jnp.ndarray)]
            
            if len(encoder_old_arrays) > 0 and len(encoder_old_arrays) == len(encoder_new_arrays):
                encoder_changes = [jnp.sqrt(jnp.mean((new - old)**2)) 
                                 for old, new in zip(encoder_old_arrays, encoder_new_arrays)]
                encoder_mean_change = jnp.mean(jnp.array(encoder_changes))
                encoder_max_change = jnp.max(jnp.array(encoder_changes))
                encoder_params_changed = jnp.sum(jnp.array(encoder_changes) > 1e-10)
                encoder_total = len(encoder_changes)
                
                # Absolute metrics
                info["encoder_param_change_mean"] = encoder_mean_change
                info["encoder_param_change_max"] = encoder_max_change
                info["encoder_params_arrays_changed"] = encoder_params_changed
                info["encoder_total_param_arrays"] = encoder_total
                
                # Percentage metrics (for easy plotting)
                info["encoder_percent_arrays_updated"] = 100.0 * encoder_params_changed / encoder_total
        
        # Track MMDiT parameters (if using MMDiT)
        if actor_key in old_params and 'mmdit' in old_params[actor_key]:
            mmdit_old = jax.tree_util.tree_leaves(old_params[actor_key]['mmdit'])
            mmdit_new = jax.tree_util.tree_leaves(new_params[actor_key]['mmdit'])
            mmdit_old_arrays = [x for x in mmdit_old if isinstance(x, jnp.ndarray)]
            mmdit_new_arrays = [x for x in mmdit_new if isinstance(x, jnp.ndarray)]
            
            if len(mmdit_old_arrays) > 0 and len(mmdit_old_arrays) == len(mmdit_new_arrays):
                mmdit_changes = [jnp.sqrt(jnp.mean((new - old)**2)) 
                               for old, new in zip(mmdit_old_arrays, mmdit_new_arrays)]
                mmdit_mean_change = jnp.mean(jnp.array(mmdit_changes))
                mmdit_max_change = jnp.max(jnp.array(mmdit_changes))
                mmdit_params_changed = jnp.sum(jnp.array(mmdit_changes) > 1e-10)
                mmdit_total = len(mmdit_changes)
                
                # Absolute metrics
                info["mmdit_param_change_mean"] = mmdit_mean_change
                info["mmdit_param_change_max"] = mmdit_max_change
                info["mmdit_params_arrays_changed"] = mmdit_params_changed
                info["mmdit_total_param_arrays"] = mmdit_total
                
                # Percentage metrics (for easy plotting)
                info["mmdit_percent_arrays_updated"] = 100.0 * mmdit_params_changed / mmdit_total
                
                # Flag if MMDiT params are NOT changing (use jnp.where for JIT compatibility)
                info["WARNING_mmdit_frozen"] = jnp.where(mmdit_mean_change < 1e-10, 1.0, 0.0)
        else:
            # Log that MMDiT params are not found (if use_mmdit=True, this is a problem!)
            info["mmdit_in_params"] = 0.0
            info["mmdit_percent_arrays_updated"] = 0.0
        
        # Comparison metrics (if both encoder and MMDiT present)
        if 'encoder_percent_arrays_updated' in info and 'mmdit_percent_arrays_updated' in info:
            # Overall update percentage across all actor parameters
            if 'encoder_total_param_arrays' in info and 'mmdit_total_param_arrays' in info:
                total_arrays = info["encoder_total_param_arrays"] + info["mmdit_total_param_arrays"]
                total_changed = info["encoder_params_arrays_changed"] + info["mmdit_params_arrays_changed"]
                info["actor_total_percent_arrays_updated"] = 100.0 * total_changed / total_arrays

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # log learning rates
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax", "return_attention_weights")) ## Train is automatically false here since .apply_fn is not passed train arg
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey = None,
        temperature: float = 1.0,
        argmax: bool = False,
        clip_sampler: bool = True,
        return_attention_weights: bool = False,
    ) -> jnp.ndarray:
        assert len(observations["image"].shape) > 3, "Must use observation histories"

        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            actor_output = self.state.apply_fn(
                {"params": self.state.target_params},
                (observations, goals),
                current_x,
                input_time,
                train=False,  # CRITICAL: Use eval mode during sampling
                return_attention_weights=return_attention_weights,
                name="actor",
            )
            
            # Handle potential attention weights return
            if return_attention_weights and isinstance(actor_output, tuple):
                eps_pred, attn_weights = actor_output
            else:
                eps_pred = actor_output
                attn_weights = None

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hats"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if clip_sampler:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            if return_attention_weights:
                return (current_x, rng), attn_weights
            else:
                return (current_x, rng), ()

        key, rng = jax.random.split(seed)

        if len(observations["image"].shape) == 4:
            # unbatched input from evaluation
            batch_size = 1
            observations = jax.tree_map(lambda x: x[None], observations)
            goals = jax.tree_map(lambda x: x[None], goals)
        else:
            batch_size = observations["image"].shape[0]

        input_tuple, scan_output = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        for _ in range(self.config["repeat_last_step"]):
            input_tuple, last_output = fn(input_tuple, 0)
            if return_attention_weights:
                scan_output = last_output

        action_0, rng = input_tuple

        if batch_size == 1:
            # this is an evaluation call so unbatch
            if return_attention_weights:
                return action_0[0], scan_output[-1, 0]
            else:
                return action_0[0]
        else:
            if return_attention_weights:
                return action_0, scan_output[-1]
            else:
                return action_0

    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(
            observations=batch["observations"], goals=batch["goals"], seed=seed
        )

        metrics = {"mse": ((actions - batch["actions"]) ** 2).sum((-2, -1)).mean()}

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        language_conditioned: bool = False,
        imagination_augmented: bool = False,
        use_attention_in_imagination: bool = False,
        attention_type_in_imagination: str = "single_head",  # "single_head" or "multi_head"
        mmdit_type: str = None, # "mmdit"/ "incontext"
        mmdit_network_kwargs: dict = {
                "depth": 2,
                "dim_cond": 256,
                "dim_head": 64,
                "timestep_embed_dim":64,
                "heads": 4,
                "ff_mult": 4
        },
        incontext_mmdit_network_kwargs: dict = {
                "depth": 12,
                "dim_head": 128,
                "heads": 4,
                "ff_mult": 4,
                "timestep_embed_dim":256,
                "num_blocks": 3,
                "dropout_rate": 0.1,
                "hidden_dim": 256,
                "use_layer_norm": True,
        },
        embed_lang_in_observation: bool = True,
        num_similar_instructions_used: int = 2,
        use_null_similar_images: bool = False,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Algorithm config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
    ):
        assert len(actions.shape) > 1, "Must use action chunking"
        assert len(observations["image"].shape) > 3, "Must use observation histories"

        if language_conditioned:
            if not imagination_augmented:
                # Use language conditioning wrapper
                print(">>> Using LCEncodingWrapper (no imagination)")
                encoder_def = LCEncodingWrapper(
                    encoder=encoder_def,
                    use_proprio=use_proprio,
                    stop_gradient=False,
                )
            else:
                if mmdit_type is not None:
                    print(f">>> Using MMDiTScoreActor {mmdit_type}")
                    encoder_def = DiTEncoder(encoder=encoder_def, num_similar_instructions_used=num_similar_instructions_used, embed_lang_in_observation=embed_lang_in_observation)
                elif use_attention_in_imagination:
                    if attention_type_in_imagination == "single_head":
                        print(f">>> Using Imaginations_LCEncodingWrapper_Attention (use_attention_in_imagination={use_attention_in_imagination}, num_similar={num_similar_instructions_used})")
                        encoder_def = Imaginations_LCEncodingWrapper_Attention(
                            encoder=encoder_def,
                            use_proprio=use_proprio,
                            stop_gradient=False,
                            num_similar_instructions_used=num_similar_instructions_used,
                            use_null_similar_images=use_null_similar_images,
                        )
                    elif attention_type_in_imagination == "multi_head":
                        print(f">>> Using Imaginations_LCEncodingWrapper_MultiLayerMultiHeadAttention (use_attention_in_imagination={use_attention_in_imagination}, num_similar={num_similar_instructions_used})")
                        encoder_def = Imaginations_LCEncodingWrapper_MultiLayerMultiHeadAttention(
                            encoder=encoder_def,
                            use_proprio=use_proprio,
                            stop_gradient=False,
                            num_similar_instructions_used=num_similar_instructions_used
                        )
                else:
                    print(f">>> Using Imaginations_LCEncodingWrapper (no attention, num_similar={num_similar_instructions_used})")
                    encoder_def = Imaginations_LCEncodingWrapper(
                        encoder=encoder_def,
                        use_proprio=use_proprio,
                        stop_gradient=False,
                        num_similar_instructions_used=num_similar_instructions_used
                    )    
        else:
            # Use goal image conditioning wrapper
            if early_goal_concat:
                # passing None as the goal encoder causes early goal concat
                goal_encoder_def = None
            else:
                if shared_goal_encoder:
                    goal_encoder_def = encoder_def
                else:
                    goal_encoder_def = copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        
        if mmdit_type is None:
            ## ScoreActor also now takes train in __call__ and uses dropout accordingly
            # lang embedding = 512
            networks = {
                "actor": ScoreActor(
                    encoder_def,
                    FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                    MLP(
                        (
                            2 * score_network_kwargs["time_dim"],
                            score_network_kwargs["time_dim"],
                        )
                    ),
                    MLPResNet(
                        score_network_kwargs["num_blocks"],
                        actions.shape[-2] * actions.shape[-1],
                        dropout_rate=score_network_kwargs["dropout_rate"],
                        use_layer_norm=score_network_kwargs["use_layer_norm"],
                    ),
                )
            }
        else:
            rng, mmdit_rng = jax.random.split(rng)
            if mmdit_type == "mmdit":
                mmdit = MMDiT(
                    **mmdit_network_kwargs,
                    dim_modalities=(1024, actions.shape[-2] * actions.shape[-1]),  # (obs_dim, action_dim)
                    dim_outs=(1024, actions.shape[-2] * actions.shape[-1]),  # Output dimensions
                    key=mmdit_rng,
                )
                
                # Separate MMDiT into trainable parameters (arrays) and static structure
                # This allows Flax to manage the parameters while keeping the structure intact
                mmdit_params, mmdit_static = eqx.partition(mmdit, eqx.is_array)
                
                networks = {
                    "actor": MMDiTScoreActor(
                        encoder=encoder_def,
                        mmdit_params_template=mmdit_params,
                        mmdit_static=mmdit_static
                    )
                }
            elif mmdit_type == "incontext":
                mmdit = InContextMMDiT(
                    depth=incontext_mmdit_network_kwargs['depth'],
                    dim_modalities=(1024,),  # Single modality tuple (image_dim, language_dim)
                    dim_outs=(512,),  # Output dimensions 
                    dim_head=incontext_mmdit_network_kwargs["dim_head"],
                    heads=incontext_mmdit_network_kwargs["heads"],
                    ff_mult=incontext_mmdit_network_kwargs["ff_mult"],
                    key=mmdit_rng,
                )
                mmdit_params, mmdit_static = eqx.partition(mmdit, eqx.is_array)
                
                networks = {
                    "actor": InContextMMDiTScoreActor(
                            encoder_def,
                            FourierFeatures(incontext_mmdit_network_kwargs["timestep_embed_dim"], learnable=False),
                            MLP(
                                (
                                    incontext_mmdit_network_kwargs["timestep_embed_dim"],
                                    incontext_mmdit_network_kwargs["timestep_embed_dim"],
                                )
                            ),
                            MLPResNet(
                                incontext_mmdit_network_kwargs["num_blocks"],
                                actions.shape[-2] * actions.shape[-1],
                                dropout_rate=incontext_mmdit_network_kwargs["dropout_rate"],
                                use_layer_norm=incontext_mmdit_network_kwargs["use_layer_norm"],
                                hidden_dim=incontext_mmdit_network_kwargs["hidden_dim"],
                            ),
                            mmdit_params_template=mmdit_params,
                            mmdit_static=mmdit_static
                    )
                }

        model_def = ModuleDict(networks)

        rng, init_rng, dropout_rng = jax.random.split(rng, 3)
        if len(actions.shape) == 3:
            example_time = jnp.zeros((actions.shape[0], 1))
        else:
            example_time = jnp.zeros((1,))
        params = model_def.init(
            {"params": init_rng, "dropout": dropout_rng}, 
            actor=[(observations, goals), actions, example_time]
        )["params"]

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {"actor": lr_schedule}
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Log parameter structure at initialization (check state.params, not params)
        print("\n" + "="*80)
        print("AGENT INITIALIZATION - PARAMETER STRUCTURE")
        print("="*80)
        
        # Check state.params (this is what's actually used during training)
        check_params = state.params
        print(f"DEBUG: state.params type = {type(check_params)}")
        print(f"DEBUG: state.params keys = {list(check_params.keys()) if hasattr(check_params, 'keys') else 'no keys method'}")
        
        # Try to find actor params - might be at different nesting levels
        actor_params = None
        if hasattr(check_params, 'keys'):
            if 'actor' in check_params:
                actor_params = check_params['actor']
                print(f"✅ Found 'actor' at state.params['actor']")
            elif 'modules_actor' in check_params:
                # ModuleDict wraps with 'modules_' prefix
                actor_params = check_params['modules_actor']
                print(f"✅ Found 'actor' at state.params['modules_actor']")
            elif 'params' in check_params and 'actor' in check_params['params']:
                actor_params = check_params['params']['actor']
                print(f"✅ Found 'actor' at state.params['params']['actor']")
            else:
                print(f"⚠️  'actor' not found in expected locations")
        
        if actor_params is not None:
            actor_keys = list(actor_params.keys()) if hasattr(actor_params, 'keys') else []
            print(f"   Actor keys: {actor_keys}")
            
            # Check encoder
            if 'encoder' in actor_params:
                encoder_leaves = jax.tree_util.tree_leaves(actor_params['encoder'])
                encoder_arrays = [x for x in encoder_leaves if isinstance(x, jnp.ndarray)]
                encoder_count = sum(x.size for x in encoder_arrays)
                print(f"   ✅ Encoder: {len(encoder_arrays)} arrays, {encoder_count:,} total params")
            
            # Check MMDiT
            if mmdit_type is not None:
                if 'mmdit' in actor_params:
                    mmdit_leaves = jax.tree_util.tree_leaves(actor_params['mmdit'])
                    mmdit_arrays = [x for x in mmdit_leaves if isinstance(x, jnp.ndarray)]
                    mmdit_count = sum(x.size for x in mmdit_arrays)
                    print(f"   ✅ MMDiT: {len(mmdit_arrays)} arrays, {mmdit_count:,} total params")
                    
                    if mmdit_count > 1_000_000:
                        print(f"   ✅ MMDiT size looks correct ({mmdit_count/1e6:.1f}M params)")
                    else:
                        print(f"   ⚠️  MMDiT seems small ({mmdit_count:,} params)")
                else:
                    print(f"   ❌ ERROR: use_mmdit=True but 'mmdit' NOT in actor params!")
                    print(f"   This means MMDiT parameters are NOT being trained!")
                    print(f"   Available keys: {actor_keys}")
            else:
                print(f"   use_mmdit=False, not using MMDiT")
        else:
            print(f"❌ Could not find 'actor' in state.params structure!")
            print(f"   Top-level keys: {list(check_params.keys()) if hasattr(check_params, 'keys') else 'N/A'}")
        print("="*80 + "\n")

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        config = flax.core.FrozenDict(
            dict(
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
                action_dim=actions.shape[-2:],
                action_max=2.0,
                action_min=-2.0,
                betas=betas,
                alphas=alphas,
                alpha_hats=alpha_hat,
                diffusion_steps=diffusion_steps,
                action_samples=action_samples,
                repeat_last_step=repeat_last_step,
            )
        )
        return cls(state, config, lr_schedules)
