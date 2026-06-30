# file: data_loader.py
# -*- coding: utf-8 -*-
"""
Data domain: loading, preprocessing, and dataset creation.

Contains the ChiscoDataset and ChiscoSubset classes, along with all
utilities for loading data from PKL files, converting labels to
meta-classes, and creating stratified cross-validation splits.
"""

# =============================================================================
# Standard Libraries
# =============================================================================
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# Third-Party Libraries
# =============================================================================
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from torch.utils.data import Dataset

# =============================================================================
# Local Imports
# =============================================================================
from config import EPSILON, JSON_DIR
from utils import _ensure_numpy_pickle_compat

# =============================================================================
# Subject ID Mapping (for embeddings)
# =============================================================================


def create_subject_mapping(
    samples: List[Dict],
    indices: Optional[np.ndarray] = None
) -> Dict[str, int]:
    """
    Description:
    ---------------
        Creates a subject_id -> integer index mapping for the embedding
        layer. Required to convert string subject identifiers into numeric
        indices expected by the neural network embedding layer.

    Args:
    ---------------
        samples: List of all data samples (dictionaries with the 'subject'
            key).
        indices: Optional subset indices used to build the mapping.

    Returns:
    ---------------
        Dict[str, int]: Mapping from subject ID to subject index.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> samples = [{'subject': 'S1'}, {'subject': 'S2'}]
        >>> create_subject_mapping(samples)
        {'S1': 0, 'S2': 1}
    """
    if indices is None:
        unique_subjects = sorted(set(s['subject'] for s in samples))
    else:
        unique_subjects = sorted(set(samples[int(i)]['subject'] for i in indices))
    return {subject_id: idx for idx, subject_id in enumerate(unique_subjects)}


# =============================================================================
# Dataset classes
# =============================================================================


class ChiscoDataset(Dataset):
    """
    Description:
    ---------------
        PyTorch Dataset class for imagined-speech EEG data from the
        Chisco dataset. Handles normalization, augmentation, and tensor
        preparation.

    Args:
    ---------------
        samples: List of data samples.
        transform: Data augmentation function (optional).
        normalize: Normalization strategy ('zscore', 'minmax', etc.).
        augment_prob: Probability of applying augmentation.
        norm_stats: Normalization statistics (mean, std).
        exclude_channels: List of channel indices to exclude.
        subject_mapping: Subject mapping for embeddings.

    Returns:
    ---------------
        Dataset object ready for DataLoader iteration.

    Raises:
    ---------------
        ValueError: If the selected normalization strategy requires
            norm_stats but they are not provided.

    Examples:
    ---------------
        >>> dataset = ChiscoDataset(samples, normalize='zscore')
        >>> len(dataset)
        1000
    """

    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Callable] = None,
        normalize: Optional[str] = 'zscore',
        augment_prob: float = 0.5,
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
        exclude_channels: Optional[List[int]] = None,
        subject_mapping: Optional[Dict[str, int]] = None,
        unknown_subject_index: int = -1,
    ):
        self.samples = samples
        self.transform = transform
        self.normalize = normalize
        self.augment_prob = augment_prob
        self.norm_stats = norm_stats
        # Initialize with an empty list when None to simplify logic.
        self.exclude_channels = exclude_channels if exclude_channels else []
        self.subject_mapping = subject_mapping if subject_mapping else {}
        self.unknown_subject_index = unknown_subject_index

    def __len__(self) -> int:
        """
        Description:
        ---------------
            Returns the number of samples in the dataset.

        Returns:
        ---------------
            int: Number of items.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Retrieves and processes one sample by index. Applies
            normalization, channel exclusion, and augmentation.

        Args:
        ---------------
            idx: Sample index.

        Returns:
        ---------------
            Dict[str, Any]: Dictionary with EEG, label, subject, and text
            tensors/fields.

        Raises:
        ---------------
            ValueError: If normalization statistics are not found.
        """
        sample = self.samples[idx]
        eeg = sample['eeg']
        label = sample['label']

        # Remove the extra dimension if present (batch dim = 1).
        if eeg.ndim == 3 and eeg.shape[0] == 1:
            eeg = eeg.squeeze(0)

        eeg = eeg.astype(np.float32)

        # Exclude noisy or inactive channels.
        if self.exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[self.exclude_channels] = False
            eeg = eeg[keep_mask, :]

        # Apply the selected normalization strategy.
        if self.normalize == 'zscore':
            mean = eeg.mean(axis=-1, keepdims=True)
            std = eeg.std(axis=-1, keepdims=True) + EPSILON
            eeg = (eeg - mean) / std
        elif self.normalize == 'minmax':
            min_val = eeg.min(axis=-1, keepdims=True)
            max_val = eeg.max(axis=-1, keepdims=True)
            eeg = (eeg - min_val) / (max_val - min_val + EPSILON)
        elif self.normalize == 'zscore_dataset_channel':
            if self.norm_stats is None:
                raise ValueError(
                    "`norm_stats` are required for 'zscore_dataset_channel'."
                )
            mean_c = self.norm_stats['mean'].reshape(-1, 1)
            std_c = (self.norm_stats['std'].reshape(-1, 1) + EPSILON)
            eeg = (eeg - mean_c) / std_c
        elif self.normalize == 'zscore_subject_channel':
            # Subject-level normalization (legacy method, overfitting risk).
            if self.norm_stats is None:
                raise ValueError(
                    "`norm_stats` are required for 'zscore_subject_channel'."
                )

            subject_id = sample['subject']
            if subject_id not in self.norm_stats:
                raise ValueError(
                    f"Subject '{subject_id}' was not found in norm_stats."
                )

            mean_c, std_c = self.norm_stats[subject_id]
            mean_c = mean_c.reshape(-1, 1)
            std_c = (std_c.reshape(-1, 1) + EPSILON)
            eeg = (eeg - mean_c) / std_c
        elif self.normalize == 'zscore_hybrid':
            # Hybrid normalization: subject-level centering and
            # dataset-level scaling.
            if self.norm_stats is None:
                raise ValueError(
                    "`norm_stats` are required for 'zscore_hybrid'."
                )

            subject_id = sample['subject']
            if ('mean_per_subject' not in self.norm_stats or
                    'std_global' not in self.norm_stats):
                raise ValueError(
                    "norm_stats must contain 'mean_per_subject' and "
                    "'std_global'."
                )

            # If the subject is missing from train (LOSO/novel), use zero
            # centering.
            if subject_id in self.norm_stats['mean_per_subject']:
                mean_c = self.norm_stats['mean_per_subject'][subject_id]
            else:
                mean_c = np.zeros(eeg.shape[0], dtype=np.float32)
            std_global = self.norm_stats['std_global']

            mean_c = mean_c.reshape(-1, 1)
            std_global = (std_global.reshape(-1, 1) + EPSILON)

            # Step 1: Subject-wise centering
            eeg_centered = eeg - mean_c
            # Step 2: Global scaling
            eeg = eeg_centered / std_global

        # Apply augmentation with the configured probability.
        if self.transform and np.random.rand() < self.augment_prob:
            eeg = self.transform(eeg)

        eeg_tensor = torch.from_numpy(eeg.copy())
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Convert subject_id to an integer for the embedding layer.
        subject_id = sample['subject']
        # Unknown subjects are explicit. The model decides whether to reject
        # them or replace their embedding by a configured fallback.
        subject_id_int = self.subject_mapping.get(
            subject_id,
            self.unknown_subject_index
        )

        return {
            'eeg': eeg_tensor,
            'label': label_tensor,
            'subject': subject_id,
            'subject_id': torch.tensor(subject_id_int, dtype=torch.long),
            'text': sample['text']
        }


class ChiscoSubset(Dataset):
    """
    Description:
    ---------------
        Represents a subset of the main `ChiscoDataset`. Used to create
        train/val splits without copying data.

    Args:
    ---------------
        dataset: Source dataset.
        indices: Array of indices to select.

    Returns:
    ---------------
        Dataset object containing only the specified indices.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> subset = ChiscoSubset(dataset, np.array([0, 1, 2]))
        >>> len(subset)
        3
    """

    def __init__(self, dataset: ChiscoDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices
        # Cache samples for fast access.
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self) -> int:
        """Returns the subset size."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Returns an item by index within the subset.

        Args:
        ---------------
            idx: Index within the subset.

        Returns:
        ---------------
            Dict[str, Any]: Sample with an added sample_id.
        """
        item = self.dataset[self.indices[idx]]
        # Add the source sample identifier for logging.
        item['sample_id'] = int(self.indices[idx])
        return item


# =============================================================================
# Data loading and processing functions
# =============================================================================


def load_class_mapping(json_path: Path = JSON_DIR) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Description:
    ---------------
        Loads class mappings from JSON files. Reads class names and
        text-to-class mapping.

    Args:
    ---------------
        json_path: Path to the directory with JSON files.

    Returns:
    ---------------
        Tuple[Dict, Dict]: (class_names, text_to_class).

    Raises:
    ---------------
        FileNotFoundError: If files are not found.
        json.JSONDecodeError: If the JSON format is invalid.

    Examples:
    ---------------
        >>> names, mapping = load_class_mapping()
        >>> len(mapping)
        39
    """
    with open(json_path / "classnumber.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    with open(json_path / "textmaps.json", "r", encoding="utf-8") as f:
        text_to_class = json.load(f)
    return class_names, text_to_class


def load_all_data(
    data_dir: Path,
    subject_ids: List[str],
    task: str = "imagine",
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Description:
    ---------------
        Loads all PKL files for the specified subjects and task.
        Iterates through subject directories and parses EEG files.

    Args:
    ---------------
        data_dir: Root data directory.
        subject_ids: List of subject identifiers to load.
        task: Task type (for example, "imagine").
        verbose: Whether to print progress to the console.

    Returns:
    ---------------
        List[Dict]: List of loaded samples.

    Raises:
    ---------------
        No explicit exceptions (errors are logged).

    Examples:
    ---------------
        >>> data = load_all_data(Path('./data'), ['S1'], verbose=False)
        >>> len(data)
        100
    """
    if verbose:
        print(f"Loading data for subjects: {subject_ids}, task: {task}")
    _, text_to_class = load_class_mapping()
    all_samples = []
    for subject_id in subject_ids:
        subject_dir = data_dir / subject_id / "eeg"
        if not subject_dir.exists():
            if verbose:
                print(f"⚠️  {subject_id}: directory not found, skipping.")
            continue
        pkl_files = sorted(subject_dir.glob(f"*task-{task}*.pkl"))
        if verbose:
            print(f"  {subject_id}: found {len(pkl_files)} files.")
        for pkl_file in pkl_files:
            # Parse the run ID from the file name.
            run_id = pkl_file.stem.split("_run-")[1].split("_")[0]
            _ensure_numpy_pickle_compat()
            try:
                with open(pkl_file, "rb") as f:
                    data_list = pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                if verbose:
                    print(
                        f"⚠️  {subject_id}: failed to read "
                        f"{pkl_file.name}: {type(exc).__name__}: {exc}. "
                        "File skipped."
                    )
                continue
            for item in data_list:
                text, eeg = item['text'], item['input_features']
                label = text_to_class.get(text, -1)
                if label == -1:
                    if verbose:
                        print(f"⚠️  Text '{text}' was not found in textmaps.json.")
                    continue
                all_samples.append({
                    'subject': subject_id,
                    'run': run_id,
                    'text': text,
                    'label': int(label),
                    'eeg': eeg
                })
    if verbose:
        print(f"✅ Total loaded samples: {len(all_samples)}")
    return all_samples


def load_metaclass_mapping(json_path: Path = JSON_DIR) -> Tuple[Dict[str, str], Dict[int, int]]:
    """
    Description:
    ---------------
        Loads the mapping from 39 source classes to 8 meta-classes.
        Used for higher-level classification tasks.

    Args:
    ---------------
        json_path: Path to the directory with JSON files.

    Returns:
    ---------------
        Tuple[Dict, Dict]: (metaclass_names, class_to_meta).

    Raises:
    ---------------
        FileNotFoundError: If metaclasses.json is not found.

    Examples:
    ---------------
        >>> names, mapping = load_metaclass_mapping()
        >>> len(names)
        8
    """
    metaclass_file = json_path / "metaclasses.json"
    with open(metaclass_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    metaclass_names = data['metaclass_names']
    class_to_meta = {int(k): v for k, v in data['metaclass_mapping'].items()}
    return metaclass_names, class_to_meta


def load_all_data_metaclass(
    data_dir: Path,
    subject_ids: List[str],
    task: str = "imagine",
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Description:
    ---------------
        Loads all data and converts labels to 8 meta-classes. Wrapper
        around load_all_data with additional label conversion.

    Args:
    ---------------
        data_dir: Root data directory.
        subject_ids: List of subject identifiers.
        task: Task type.
        verbose: Information output flag.

    Returns:
    ---------------
        List[Dict]: List of samples with updated labels.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> data = load_all_data_metaclass(Path('./data'), ['S1'])
        >>> data[0]['label']
        3
    """
    samples = load_all_data(data_dir, subject_ids, task, verbose)
    _, class_to_meta = load_metaclass_mapping()
    converted_samples = []
    for sample in samples:
        orig_label = sample['label']
        meta_label = class_to_meta.get(orig_label, -1)
        if meta_label != -1:
            converted_sample = sample.copy()
            converted_sample['label'] = meta_label
            converted_sample['original_label'] = orig_label
            converted_samples.append(converted_sample)
    if verbose:
        print("\n" + "=" * 80)
        print(f"Meta-class conversion: {len(converted_samples)} samples.")
        print("=" * 80 + "\n")
    return converted_samples


def get_stratified_cv_splits(
    labels: np.ndarray,
    n_splits: int,
    random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
    ---------------
        Creates stratified K-Fold splits for cross-validation.
        Preserves class balance in each fold.

    Args:
    ---------------
        labels: Class label array.
        n_splits: Number of splits (K).
        random_state: Seed for reproducibility.

    Returns:
    ---------------
        List[Tuple]: List of tuples (train_idx, val_idx).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> splits = get_stratified_cv_splits(np.array([0, 1, 0, 1]), 2, 42)
        >>> len(splits)
        2
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(skf.split(np.zeros(len(labels)), labels))


def get_within_subject_cv_splits(
    samples: List[Dict],
    labels: np.ndarray,
    n_splits: int,
    random_state: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
    ---------------
        Creates within-subject K-Fold splits: each subject is split
        separately with StratifiedKFold, then folds are merged across
        subjects. This guarantees that each subject is present in both
        train and val, and that trainable subject embeddings have training
        observations before validation.

    Args:
    ---------------
        samples: List of all data samples.
        labels: Class label array.
        n_splits: Number of folds.
        random_state: Seed for reproducibility.

    Returns:
    ---------------
        List[Tuple]: List of tuples (train_idx, val_idx).

    Raises:
    ---------------
        ValueError: If a subject has insufficient data for stratified split.
    """
    subject_to_indices: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        subject_id = sample['subject']
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)

    combined: List[Tuple[List[int], List[int]]] = [
        ([], []) for _ in range(n_splits)
    ]

    for subject_id, subject_indices_list in sorted(subject_to_indices.items()):
        subject_indices = np.asarray(subject_indices_list, dtype=int)
        subject_labels = labels[subject_indices]
        _, class_counts = np.unique(subject_labels, return_counts=True)

        if len(subject_indices) < n_splits or np.min(class_counts) < n_splits:
            raise ValueError(
                f"Subject '{subject_id}' does not support within_subject "
                f"StratifiedKFold: n_samples={len(subject_indices)}, "
                f"min_class_count={int(np.min(class_counts))}, "
                f"n_splits={n_splits}."
            )

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        for fold_idx, (local_train, local_val) in enumerate(
            skf.split(np.zeros(len(subject_labels)), subject_labels)
        ):
            train_acc, val_acc = combined[fold_idx]
            train_acc.extend(subject_indices[local_train].tolist())
            val_acc.extend(subject_indices[local_val].tolist())

    return [
        (
            np.asarray(train_idx, dtype=int),
            np.asarray(val_idx, dtype=int)
        )
        for train_idx, val_idx in combined
    ]


def get_stratified_group_cv_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
    ---------------
        Stratified K-Fold grouped by subject. Guarantees that the same
        group (subject) is not present in train and val at the same time.
        Critical for evaluating cross-subject generalization.

    Args:
    ---------------
        labels: np.ndarray [N] - Meta-classes (0..7).
        groups: np.ndarray [N] - subject_id indices (integers).
        n_splits: int - Number of folds (default: 5).
        random_state: int - Seed for reproducibility.

    Returns:
    ---------------
        List[Tuple]: Stratified grouped splits.

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> samples = load_all_data_metaclass(...)
        >>> labels = np.array([s['label'] for s in samples])
        >>> subject_mapping = create_subject_mapping(samples)
        >>> groups = np.array([subject_mapping[s['subject']] for s in samples])
        >>> splits = get_stratified_group_cv_splits(labels, groups, n_splits=5)
    """
    sgf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(sgf.split(X=np.arange(len(labels)), y=labels, groups=groups))


def get_loso_splits(
    samples: List[Dict],
    subject_mapping: Dict[str, int]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
    ---------------
        Leave-One-Subject-Out cross-validation. For each subject:
        Test contains all samples from that subject, and Train contains
        all samples from the remaining subjects. This is the strictest
        evaluation.

    Args:
    ---------------
        samples: List[Dict] - List of all data samples.
        subject_mapping: Dict[str, int] - subject_id -> index mapping.

    Returns:
    ---------------
        List[Tuple]: LOSO splits (train_idx, val_idx).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> splits = get_loso_splits(samples, subject_mapping)
        >>> print(f"LOSO: {len(splits)} splits for {len(subject_mapping)} subjects")
    """
    unique_subjects = sorted(subject_mapping.keys())
    splits = []

    for test_subject in unique_subjects:
        # Mask for selecting the test subject.
        test_mask = np.array([s['subject'] == test_subject for s in samples])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))

    return splits


def compute_channelwise_stats(
    samples: List[Dict],
    indices: np.ndarray,
    exclude_channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
    ---------------
        Computes the mean and standard deviation for each channel.
        Used for global dataset normalization.

    Args:
    ---------------
        samples: List of all data samples.
        indices: Sample indices used to compute statistics.
        exclude_channels: List of channel indices to exclude.

    Returns:
    ---------------
        Tuple[np.ndarray, np.ndarray]: (mean_c, std_c).

    Raises:
    ---------------
        ValueError: If the index array is empty.

    Examples:
    ---------------
        >>> mean, std = compute_channelwise_stats(samples, np.array([0, 1]))
        >>> mean.shape
        (n_channels,)
    """
    exclude_channels = exclude_channels or []
    if len(indices) == 0:
        raise ValueError("The index array cannot be empty.")

    eeg0 = samples[indices[0]]['eeg'].squeeze(0)
    if exclude_channels:
        keep_mask = np.ones(eeg0.shape[0], dtype=bool)
        keep_mask[exclude_channels] = False
        eeg0 = eeg0[keep_mask, :]
    C, _ = eeg0.shape

    sum_c, sumsq_c, count = (
        np.zeros(C, np.float64),
        np.zeros(C, np.float64),
        0
    )
    for idx in indices:
        eeg = samples[idx]['eeg'].squeeze(0)
        if exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[exclude_channels] = False
            eeg = eeg[keep_mask, :]
        x = eeg.astype(np.float64)
        sum_c += x.sum(axis=-1)
        sumsq_c += (x ** 2).sum(axis=-1)
        count += x.shape[1]

    mean_c = sum_c / count
    var_c = (sumsq_c / count) - (mean_c ** 2)
    # Protect against negative variance due to rounding errors.
    std_c = np.sqrt(np.maximum(var_c, 1e-12))
    return mean_c.astype(np.float32), std_c.astype(np.float32)


def compute_subjectwise_stats(
    samples: List[Dict],
    indices: np.ndarray,
    exclude_channels: Optional[List[int]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Description:
    ---------------
        Computes mean/std for each subject separately. Groups samples by
        subject_id and computes channelwise statistics for each subject
        independently. Removes subject baseline shifts (impedance, skull
        thickness).

    Args:
    ---------------
        samples: List of all data samples.
        indices: Sample indices for the training split.
        exclude_channels: List of channel indices to exclude.

    Returns:
    ---------------
        Dict[str, Tuple]: subject_id -> (mean_c, std_c).

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> stats = compute_subjectwise_stats(samples, train_indices)
        >>> 'S1' in stats
        True
    """
    exclude_channels = exclude_channels or []

    # Group indices by subject
    subject_groups = {}
    for idx in indices:
        subject_id = samples[idx]['subject']
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(idx)

    # Compute stats per subject
    stats = {}
    for subject_id, subject_indices in subject_groups.items():
        subject_indices_np = np.array(subject_indices)
        mean_c, std_c = compute_channelwise_stats(
            samples, subject_indices_np, exclude_channels
        )
        stats[subject_id] = (mean_c, std_c)

    return stats


def compute_hybrid_stats(
    samples: List[Dict],
    indices: np.ndarray,
    exclude_channels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Description:
    ---------------
        HYBRID NORMALIZATION: Subject-wise centering + Global scaling.
        Combination of subject-wise and global normalization to maximize
        the discriminative signal while minimizing subject baseline shifts.

    Args:
    ---------------
        samples: List of all data samples.
        indices: Sample indices for the training split.
        exclude_channels: List of channel indices to exclude.

    Returns:
    ---------------
        Dict[str, Any]:
            'mean_per_subject': Dict[subject_id -> mean_c]
            'std_global': np.ndarray (n_channels,)

    Raises:
    ---------------
        No explicit exceptions.

    Examples:
    ---------------
        >>> stats = compute_hybrid_stats(samples, train_indices)
        >>> 'std_global' in stats
        True
    """
    exclude_channels = exclude_channels or []

    # Group indices by subject
    subject_groups = {}
    for idx in indices:
        subject_id = samples[idx]['subject']
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(idx)

    # Step 1: Compute per-subject means (for centering)
    subject_means = {}
    for subject_id, subject_indices in subject_groups.items():
        subject_indices_np = np.array(subject_indices)
        mean_c, _ = compute_channelwise_stats(
            samples, subject_indices_np, exclude_channels
        )
        subject_means[subject_id] = mean_c

    # Step 2: Compute GLOBAL std on centered data
    # Accumulate centered data statistics
    eeg0 = samples[indices[0]]['eeg'].squeeze(0)
    if exclude_channels:
        keep_mask = np.ones(eeg0.shape[0], dtype=bool)
        keep_mask[exclude_channels] = False
        eeg0 = eeg0[keep_mask, :]
    C, _ = eeg0.shape

    sumsq_c_global = np.zeros(C, np.float64)
    count_global = 0

    for idx in indices:
        subject_id = samples[idx]['subject']
        eeg = samples[idx]['eeg'].squeeze(0)
        if exclude_channels:
            keep_mask = np.ones(eeg.shape[0], dtype=bool)
            keep_mask[exclude_channels] = False
            eeg = eeg[keep_mask, :]

        # Center by subject mean
        x_centered = (
            eeg.astype(np.float64) -
            subject_means[subject_id].reshape(-1, 1)
        )

        sumsq_c_global += (x_centered ** 2).sum(axis=-1)
        count_global += x_centered.shape[1]

    # Global std (on centered data)
    var_c_global = sumsq_c_global / count_global
    std_c_global = np.sqrt(np.maximum(var_c_global, 1e-12))

    return {
        'mean_per_subject': subject_means,
        'std_global': std_c_global.astype(np.float32)
    }
