import fnmatch
import sys
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging

from jaxrl_m.data.tf_augmentations import augment
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


MAX_SIMILAR_INSTRUCTIONS = 6


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        assert len(filtered_paths) > 0, f"{glob_str} came up empty"
        path_list += filtered_paths
    return path_list

class CalvinDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    Calvin dataset format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        action_proprio_metadata: Dictionary containing metadata of the actions and proprio.
            If provided, actions and proprio will be normalized.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
        load_langauge: Whether to look for and load language from the data.
        skip_unlabeled: Whether to filter out trajectories not labeled with language.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_proprio_metadata: Optional[dict] = None,
        normalization_type: Optional[str] = "normal",
        relabel_actions: bool = True,
        use_goal_relabeling: bool = True,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict = {},
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: dict = {},
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        load_language: bool = False,
        load_susie_goal_images: bool = False,
        load_img_similar_instruct: bool = False,
        skip_unlabeled: bool = False,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to CalvinDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.relabel_actions = relabel_actions
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        self.use_goal_relabeling = use_goal_relabeling
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.cache = cache
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.act_pred_horizon = act_pred_horizon
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.load_susie_goal_images = load_susie_goal_images
        self.load_img_similar_instruct = load_img_similar_instruct
        self.load_language = load_language

        if self.load_language:
            self.PROTO_TYPE_SPEC["language_annotation"] = tf.string
        if self.load_susie_goal_images or self.load_img_similar_instruct:
            self.PROTO_TYPE_SPEC["susie_goal_images"] = tf.uint8
        if self.load_img_similar_instruct:
            self.PROTO_TYPE_SPEC["similar_instructions"] = tf.string

        # construct a dataset for each sub-list of paths
        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if skip_unlabeled:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(x["goals"]["language"] != "")
            )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # even if not augmenting, we may need to concat susie goal images
            dataset = dataset.map(self.susie_concat, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)

        # yields trajectories
        #dataset = dataset.map(self._process_actions, num_parallel_calls=tf.data.AUTOTUNE) # we're skipping action normalization

        # yields trajectories
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "actions": tf.float32,     
        "proprioceptive_states": tf.float32,
        "image_states": tf.uint8,
    }
    # @tf.autograph.experimental.do_not_convert
    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: (tf.io.VarLenFeature(tf.string) if key == "similar_instructions" 
                  else tf.io.FixedLenFeature([], tf.string))
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {}
        for key, dtype in self.PROTO_TYPE_SPEC.items():
            if dtype == tf.string:
                if key == "similar_instructions":
                    # Handle sparse tensor for list of strings
                    parsed_tensors[key] = tf.sparse.to_dense(parsed_features[key], default_value=b'')
                else:
                    parsed_tensors[key] = parsed_features[key]
            else:
                parsed_tensors[key] = tf.io.parse_tensor(parsed_features[key], dtype)
        
        # Handle susie_goal_images shape based on load_img_similar_instruct
        result = {
            "observations": {
                "image": parsed_tensors["image_states"][:-1],
                "proprio": parsed_tensors["proprioceptive_states"][:-1],
            },
            "next_observations": {
                "image": parsed_tensors["image_states"][1:],
                "proprio": parsed_tensors["proprioceptive_states"][1:],
            },
            **({"language": parsed_tensors["language_annotation"]} if self.load_language else {}),
            "actions": parsed_tensors["actions"][:-1],
            "terminals": tf.zeros_like(parsed_tensors["actions"][:-1][:, 0:1], dtype=tf.bool),
        }
        
        if self.load_img_similar_instruct:
            # Shape: (num_similar_instructions, traj_len, H, W, C)
            # Slice off last timestep: (num_similar_instructions, traj_len-1, H, W, C)
            result["susie_goal_images"] = parsed_tensors["susie_goal_images"][:, :-1]
            result["similar_instructions"] = parsed_tensors["similar_instructions"]
        elif self.load_susie_goal_images:
            # Shape: (traj_len, H, W, C) - single goal image per timestep
            result["susie_goal_images"] = parsed_tensors["susie_goal_images"][:-1]
        
        return result

    def _process_actions(self, traj):
        # normalize actions and proprio
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # normalize to mean 0, std 1
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["mean"]
                ) / self.action_proprio_metadata["action"]["std"]
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["mean"]
                    ) / self.action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type == "bounds":
                # normalize to [0, 1]
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["min"]
                ) / (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
                # clip to [0, 1]
                traj["actions"] = tf.clip_by_value(traj["actions"], 0, 1)
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    ) / (
                        self.action_proprio_metadata["proprio"]["max"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    )
                    traj[key]["proprio"] = tf.clip_by_value(traj[key]["proprio"], 0, 1)
            else:
                raise ValueError

        return traj

    # @tf.autograph.experimental.do_not_convert
    def _chunk_act_obs(self, traj):
        traj_len = tf.shape(traj["actions"])[0]
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
            traj["next_obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["next_observations"]
            )
            if self.load_img_similar_instruct:
                num_similar = tf.shape(traj["susie_goal_images"])[0]
                traj_len = tf.shape(traj["susie_goal_images"])[1]
                H, W, C = tf.shape(traj["susie_goal_images"])[2], tf.shape(traj["susie_goal_images"])[3], tf.shape(traj["susie_goal_images"])[4]
                # Pad to MAX_SIMILAR_INSTRUCTIONS if needed
                num_to_pad = tf.maximum(MAX_SIMILAR_INSTRUCTIONS - num_similar, 0)
                padding = tf.zeros([num_to_pad, traj_len, H, W, C], dtype=traj["susie_goal_images"].dtype)
                padded_goals = tf.concat([traj["susie_goal_images"], padding], axis=0)
                
                lang_padding = tf.fill([num_to_pad], b'')
                padded_similar = tf.concat([traj["similar_instructions"], lang_padding],axis=0)
                lang = traj["language"]
                traj["similar_instructions"]= tf.concat([
                    lang[tf.newaxis],  # Original language instruction
                    padded_similar     # Similar instructions (with padding)
                ], axis=0)
                
                # Shape: (num_similar_instructions, traj_len, H, W, C)
                # Transpose to: (traj_len, num_similar_instructions, H, W, C)
                transposed = tf.transpose(padded_goals, [1, 0, 2, 3, 4])
                
                # Use tf.gather to chunk along traj_len dimension, same as observations
                # Result: (traj_len, obs_horizon, num_similar_instructions, H, W, C)
                traj["susie_goal_images"] = tf.gather(transposed, chunk_indices)
                valid_mask = tf.concat([
                    tf.ones([1], dtype=tf.float32),  # Original instruction is always valid
                    tf.concat([
                        tf.ones([num_similar], dtype=tf.float32),  # Real similar instructions
                        tf.zeros([num_to_pad], dtype=tf.float32)   # Padding
                    ], axis=0)
                ], axis=0)
                traj["valid_mask"] = valid_mask
            elif self.load_susie_goal_images:
                # Original single goal image case: (traj_len, H, W, C)
                # Use tf.gather to chunk along traj_len dimension
                # Result: (traj_len, obs_horizon, H, W, C)
                traj["susie_goal_images"] = tf.gather(traj["susie_goal_images"], chunk_indices)
        return traj

    # @tf.autograph.experimental.do_not_convert
    def _add_goals(self, traj):
        if self.load_language:
            lang = traj["language"]
            traj["language"] = tf.broadcast_to(
                lang, tf.shape(traj["terminals"])
            )
        
        if self. load_img_similar_instruct:
            traj["similar_instructions"] = tf.broadcast_to(
                traj["similar_instructions"], 
                [tf.shape(traj["terminals"])[0], MAX_SIMILAR_INSTRUCTIONS + 1]
            )
            traj["valid_mask"] = tf.broadcast_to(
                traj["valid_mask"], 
                [tf.shape(traj["terminals"])[0], MAX_SIMILAR_INSTRUCTIONS + 1]
            )
            
            
        if self.use_goal_relabeling:
            traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
                traj, **self.goal_relabeling_kwargs
            )
        else:
            assert "susie_goal_images" in traj and self.load_susie_goal_images, "susie_goal_images must be in traj if not using goal relabeling"
            traj["goals"] = {}
            if self.obs_horizon is not None:
                traj["goals"]["image"] = traj["susie_goal_images"][:, -1]
            else:
                traj["goals"]["image"] = traj["susie_goal_images"]
            traj_len = tf.shape(traj["goals"]["image"])[0]
                
            traj["goal_dists"] = traj_len - tf.range(traj_len)

        if self.load_language:
            # lang = traj["language"]
            traj["goals"]["language"] = tf.broadcast_to(
                traj["language"], tf.shape(traj["terminals"])
            )
            traj.pop("language")

            # Handle similar_instructions and valid_mask similarly as above for language
            if self.load_img_similar_instruct:
                traj["goals"]["language_with_similar"] = tf.broadcast_to(
                    traj["similar_instructions"], 
                    [tf.shape(traj["terminals"])[0], MAX_SIMILAR_INSTRUCTIONS + 1]
                )
                traj["goals"]["susie_goal_valid_mask"] = tf.broadcast_to(
                    traj["valid_mask"],
                    [tf.shape(traj["terminals"])[0], MAX_SIMILAR_INSTRUCTIONS + 1]
                )
                traj.pop("similar_instructions")
                traj.pop("valid_mask")
                
                # tf.print("DEBUG _add_goals: language_with_similar shape before broadcast:", 
                #         tf.shape(traj["goals"]["language_with_similar"]), 
                #         output_stream=sys.stdout)
                

            # always make the "goal" the last obs so that masking is done
            # properly below
            traj_len = tf.shape(traj["goal_dists"])[0]
            traj["goal_dists"] = traj_len - tf.range(traj_len)

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
            # set movement actions to 0 after the goal is reached
            new_movement = tf.where(
                (
                    traj["goal_dists"][:, None, None]
                    > tf.range(self.act_pred_horizon)[None, :, None]
                ),  # shape (traj_len, act_pred_horizon, 1)
                traj["actions"][
                    :, :, :-1
                ],  # shape (traj_len, act_pred_horizon, action_dim - 1)
                tf.zeros_like(traj["actions"][0, 0, :-1]),  # shape (action_dim - 1)
            )
            # for gripper actions, repeat the last action after the goal is reached
            new_gripper = tf.where(
                (
                    traj["goal_dists"][:, None]
                    > tf.range(self.act_pred_horizon)[None, :]
                ),  # shape (traj_len, act_pred_horizon)
                traj["actions"][:, :, -1],  # shape (traj_len, act_pred_horizon)
                tf.gather(
                    # shifts `actions` to the right by one, padding with the first action
                    tf.concat(
                        [
                            tf.concat(
                                [
                                    traj["actions"][:1, :1, -1],
                                    traj["actions"][:1, :-1, -1],
                                ],
                                axis=1,
                            ),
                            traj["actions"][:-1, :, -1],
                        ],
                        axis=0,
                    ),
                    # selects the action at index `goal_dists` in the previous action chunk
                    tf.minimum(traj["goal_dists"], self.act_pred_horizon - 1),
                    batch_dims=1,
                )[:, None],
            )
            traj["actions"] = tf.concat([new_movement, new_gripper[:, :, None]], axis=2)
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")
            traj["next_observations"] = traj.pop("next_obs_chunks")

        return traj
    
    # @tf.autograph.experimental.do_not_convert
    def _augment(self, seed, image):
        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [4, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            # use the same seed for obs, next_obs, and goal
            sub_seeds = [[seed, seed]] * 4

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals"], sub_seeds
        ):
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        
        # AFTER augmentation, concatenate susie goal images
        if self.load_img_similar_instruct:
            assert "susie_goal_images" in image, "susie_goal_images key missing in image dict"
            # Shape: (obs_horizon, MAX_SIMILAR_INSTRUCTIONS, H, W, C)
            obs_h = tf.shape(image["susie_goal_images"])[0]
            goal_shape = tf.shape(image["susie_goal_images"])
            H, W, C = goal_shape[2], goal_shape[3], goal_shape[4]
            

            
            # Apply same augmentation to all similar instruction goals
            # Reshape to (obs_horizon * MAX_SIMILAR_INSTRUCTIONS, H, W, C)
            reshaped_goals = tf.reshape(
                image["susie_goal_images"],
                [obs_h * MAX_SIMILAR_INSTRUCTIONS, H, W, C]
            )
            
            # Apply augmentation
            augmented_goals = augment(
                reshaped_goals, sub_seeds[-1], **self.augment_kwargs
            )
            
            # Reshape back to (obs_horizon, MAX_SIMILAR_INSTRUCTIONS, H, W, C)
            augmented_goals = tf.reshape(
                augmented_goals,
                [obs_h, MAX_SIMILAR_INSTRUCTIONS, H, W, C]
            )
            
            # Add extra dimension to observations: (obs_horizon, H, W, C) -> (obs_horizon, 1, H, W, C)
            obs_with_dim = image["observations"]["image"][:, tf.newaxis, ...]
            
            # Concatenate along axis=1 to get (obs_horizon, MAX_SIMILAR_INSTRUCTIONS+1, H, W, C)
            # First element is original observation, rest are SUSIE goals (including padding)
            image["observations"]["image"] = tf.concat([
                obs_with_dim,
                augmented_goals
            ], axis=1)
            
            # Remove the separate susie_goal_images tensor
            image.pop("susie_goal_images")
        elif self.load_susie_goal_images and self.use_goal_relabeling:
            # Original single goal image case
            # Apply same augmentation to susie goal images
            image["susie_goal_images"] = augment(
                image["susie_goal_images"], sub_seeds[-1], **self.augment_kwargs
            )
            
            # Now concatenate the augmented images
            image["observations"]["image"] = tf.concat([
                image["observations"]["image"], 
                image["susie_goal_images"]
            ], axis=-1)
            
            # Remove the separate susie_goal_images tensor
            image.pop("susie_goal_images")
        return image
    # @tf.autograph.experimental.do_not_convert
    def susie_concat(self, image):
            
        if self.load_img_similar_instruct:
            
            # Add extra dimension to observations: (obs_horizon, H, W, C) -> (obs_horizon, 1, H, W, C)
            obs_with_dim = image["observations"]["image"][:, tf.newaxis, ...]
            
            # Concatenate along axis=1 to get (obs_horizon, MAX_SIMILAR_INSTRUCTIONS+1, H, W, C)
            # First element is original observation, rest are SUSIE goals (including padding)
            image["observations"]["image"] = tf.concat([
                obs_with_dim,
                image["susie_goal_images"]
            ], axis=1)
            
            image.pop("susie_goal_images")
        elif self.load_susie_goal_images and self.use_goal_relabeling and "susie_goal_images" in image:
            # Original single goal image case
            image["observations"]["image"] = tf.concat([
                image["observations"]["image"], 
                image["susie_goal_images"]
            ], axis=-1)
            
            image.pop("susie_goal_images")
        return image

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()