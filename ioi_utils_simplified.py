#!/usr/bin/env python3
"""
Simplified IOI utilities for causal circuit discovery
Includes only the essential classes needed without external data dependencies
"""

import random
import torch
import numpy as np
from typing import Tuple, List, Union, Sequence
from collections import OrderedDict
from torch.utils.data import Dataset

def is_single_token(s: str, tokenizer) -> bool:
    """Check if a string is a single token in the vocabulary of a model."""
    return len(tokenizer.tokenize(s)) == 1

# Simplified data - subset of the full IOI dataset
NAMES = [
    "John", "Mary", "James", "Jennifer", "Robert", "Linda", "Michael", "Elizabeth", 
    "William", "Barbara", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah",
    "Thomas", "Karen", "Christopher", "Lisa", "Daniel", "Nancy", "Matthew", "Betty",
    "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna", "Steven", "Carol"
]

OBJECTS = [
    "book", "pen", "cup", "phone", "key", "wallet", "bag", "watch", "hat", "jacket"
]

PLACES = [
    "park", "store", "school", "office", "library", "cafe", "hospital", "bank", "gym", "restaurant"
]

TEMPLATES = [
    "Then, {name_A} and {name_B} had a long and really crazy argument. Afterwards, {name_C} said to",
    "Afterwards, {name_A} and {name_B} went to the {place}. {name_C} gave a {object} to",
    "When {name_A} and {name_B} got really mad, {name_C} said to",
    "After {name_A} and {name_B} went to the {place}, {name_C} gave the {object} to",
    "While {name_A} and {name_B} were arguing about the {object}, {name_C} gave it to"
]

class Prompt:
    """Represent a general ABC prompt using a template, and operations on it."""

    def __init__(self, names: Tuple[str, str, str], template: str, obj: str, place: str):
        self.names = names
        self.template = template
        self.obj = obj
        self.place = place
        if self.is_ioi:
            self.s_name = self.names[2]  # subject always appears in third position
            self.io_name = [x for x in self.names[:2] if x != self.s_name][0]
            self.s1_pos = self.names[:2].index(self.s_name)
            self.io_pos = self.names[:2].index(self.io_name)
            self.s2_pos = 2
        else:
            self.io_name = None
            self.s_name = None

    @property
    def is_ioi(self) -> bool:
        return self.names[2] in self.names[:2] and len(set(self.names)) == 2

    def __repr__(self) -> str:
        return f"<===PROMPT=== {self.sentence}>"

    @property
    def sentence(self) -> str:
        return self.template.format(
            name_A=self.names[0],
            name_B=self.names[1],
            name_C=self.names[2],
            object=self.obj,
            place=self.place,
        )

    @staticmethod
    def canonicalize(things: Tuple[str, str, str]) -> Tuple[str, str, str]:
        ordered_uniques = list(OrderedDict.fromkeys(things).keys())
        canonical_elts = ["A", "B", "C"]
        uniques_to_canonical = {
            x: y for x, y in zip(ordered_uniques, canonical_elts[: len(ordered_uniques)])
        }
        return tuple([uniques_to_canonical[x] for x in things])

    @staticmethod
    def matches_pattern(names: Tuple[str, str, str], pattern: str) -> bool:
        return Prompt.canonicalize(names) == Prompt.canonicalize(tuple(pattern))

    def resample_pattern(self, orig_pattern: str, new_pattern: str, name_distribution: Sequence[str]) -> "Prompt":
        """Change the pattern of the prompt, while keeping the names that are mapped to the same symbols."""
        assert len(orig_pattern) == 3
        assert len(new_pattern) == 3
        assert self.matches_pattern(names=self.names, pattern=orig_pattern)
        
        orig_to_name = {orig_pattern[i]: self.names[i] for i in range(3)}
        new_names = [None for _ in range(3)]
        new_pos_to_symbol = {}
        
        for i, symbol in enumerate(new_pattern):
            if symbol in orig_to_name.keys():
                new_names[i] = orig_to_name[symbol]
            else:
                new_pos_to_symbol[i] = symbol
                
        new_symbols = new_pos_to_symbol.values()
        if len(new_symbols) > 0:
            new_symbol_to_name = {}
            available_names = [x for x in name_distribution if x not in self.names]
            for symbol in new_symbols:
                new_symbol_to_name[symbol] = random.choice(available_names)
                available_names.remove(new_symbol_to_name[symbol])
            for i, symbol in new_pos_to_symbol.items():
                new_names[i] = new_symbol_to_name[symbol]
                
        return Prompt(names=tuple(new_names), template=self.template, obj=self.obj, place=self.place)


class PromptDataset(Dataset):
    def __init__(self, prompts: List[Prompt], tokenizer):
        assert len(prompts) > 0
        self.prompts: Sequence[Prompt] = np.array(prompts)
        self.tokenizer = tokenizer

    def __getitem__(self, idx: Union[int, Sequence]) -> "PromptDataset":
        if isinstance(idx, int):
            prompts = [self.prompts[idx]]
        else:
            prompts = self.prompts[idx]
            if isinstance(prompts, Prompt):
                prompts = [prompts]
        assert all(isinstance(x, Prompt) for x in prompts)
        return PromptDataset(prompts=prompts, tokenizer=self.tokenizer)

    def __len__(self) -> int:
        return len(self.prompts)

    def __add__(self, other: "PromptDataset") -> "PromptDataset":
        return PromptDataset(prompts=list(self.prompts) + list(other.prompts), tokenizer=self.tokenizer)

    @property
    def lengths(self) -> List[int]:
        return [
            self.tokenizer(x.sentence, return_tensors="pt", return_attention_mask=False)["input_ids"].shape[1]
            for x in self.prompts
        ]

    @property
    def tokens(self):
        return self.tokenizer([x.sentence for x in self.prompts], return_tensors="pt", padding=True)

    @property
    def io_tokens(self):
        return torch.tensor([self.tokenizer(f" {x.io_name}")["input_ids"][0] for x in self.prompts])

    @property
    def s_tokens(self):
        return torch.tensor([self.tokenizer(f" {x.s_name}")["input_ids"][0] for x in self.prompts])

    @property
    def answer_tokens(self):
        return torch.tensor([
            [self.tokenizer(f" {x.io_name}")["input_ids"][0], self.tokenizer(f" {x.s_name}")["input_ids"][0]]
            for x in self.prompts
        ])


class PatchingDataset(Dataset):
    """Bundle together the data needed to train patching for a single causal variable."""

    def __init__(self, base: PromptDataset, source: PromptDataset, patched_answer_tokens):
        assert len(base) == len(source)
        assert len(base) == len(patched_answer_tokens)
        self.base = base
        self.source = source
        self.patched_answer_tokens = patched_answer_tokens.long()

    def batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            order = np.random.permutation(len(self))
        else:
            order = np.arange(len(self))
        for i in range(0, len(self), batch_size):
            yield self[order[i : i + batch_size]]

    def __getitem__(self, idx) -> "PatchingDataset":
        patched_answer_tokens = self.patched_answer_tokens[idx]
        if len(patched_answer_tokens.shape) == 1:
            patched_answer_tokens = patched_answer_tokens.unsqueeze(0)
        return PatchingDataset(
            base=self.base[idx], source=self.source[idx], patched_answer_tokens=patched_answer_tokens
        )

    def __len__(self) -> int:
        return len(self.base)

    def __add__(self, other: "PatchingDataset") -> "PatchingDataset":
        return PatchingDataset(
            base=self.base + other.base,
            source=self.source + other.source,
            patched_answer_tokens=torch.cat([self.patched_answer_tokens, other.patched_answer_tokens], dim=0),
        )


class PromptDistribution:
    """A class to represent a distribution over prompts."""

    def __init__(self, names: List[str], places: List[str], objects: List[str], templates: List[str]):
        self.names = names
        self.places = places
        self.objects = objects
        self.templates = templates

    def sample_one(self, pattern: str) -> Prompt:
        """Sample a single prompt from the distribution."""
        template = random.choice(self.templates)
        unique_ids = list(set(pattern))
        unique_names = random.sample(self.names, len(unique_ids))
        assert len(set(unique_names)) == len(unique_names)
        prompt_names = tuple([unique_names[unique_ids.index(i)] for i in pattern])
        obj = random.choice(self.objects)
        place = random.choice(self.places)
        return Prompt(names=prompt_names, template=template, obj=obj, place=place)

    def sample_das(self, tokenizer, base_patterns: List[str], source_patterns: List[str], 
                   samples_per_combination: int, labels: str) -> PatchingDataset:
        """Sample a dataset of base and corrupted prompts for doing DAS."""
        base_prompts: List[Prompt] = []
        source_prompts: List[Prompt] = []
        
        for orig_pattern in base_patterns:
            for corrupted_pattern in source_patterns:
                base_prompt_batch = [self.sample_one(orig_pattern) for _ in range(samples_per_combination)]
                source_prompt_batch = [
                    p.resample_pattern(
                        name_distribution=self.names, orig_pattern=orig_pattern, new_pattern=corrupted_pattern
                    )
                    for p in base_prompt_batch
                ]
                base_prompts.extend(base_prompt_batch)
                source_prompts.extend(source_prompt_batch)

        # Generate patched answer labels
        if labels == "position":
            patched_answer_names = []
            for base_prompt, source_prompt in zip(base_prompts, source_prompts):
                if hasattr(base_prompt, 's1_pos') and hasattr(source_prompt, 's1_pos'):
                    if base_prompt.s1_pos == source_prompt.s1_pos:
                        patched_answer_names.append((base_prompt.io_name, base_prompt.s_name))
                    else:
                        patched_answer_names.append((base_prompt.s_name, base_prompt.io_name))
                else:
                    # Default behavior for IOI task
                    patched_answer_names.append((base_prompt.io_name, base_prompt.s_name))
        elif labels == "name":
            patched_answer_names = []
            for base_prompt, source_prompt in zip(base_prompts, source_prompts):
                patched_answer_names.append((source_prompt.io_name, base_prompt.io_name))

        clean_dataset = PromptDataset(base_prompts, tokenizer)
        corrupted_dataset = PromptDataset(source_prompts, tokenizer)
        patched_answer_tokens = torch.Tensor([
            [tokenizer(f" {x}")["input_ids"][0] for x in y] for y in patched_answer_names
        ])
        
        return PatchingDataset(
            base=clean_dataset, source=corrupted_dataset, patched_answer_tokens=patched_answer_tokens
        )


# Utility functions
def compute_metrics(eval_preds, eval_labels):
    """Compute accuracy and KL divergence metrics."""
    total_count = 0
    correct_count = 0
    kl_divs = []
    
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
        kl_divs += [eval_pred[:, -1][torch.arange(len(actual_test_labels)), actual_test_labels]]
        
    accuracy = round(correct_count / total_count, 2)
    kl_div = torch.cat(kl_divs, dim=0).mean()
    return {"accuracy": accuracy, "kl_div": kl_div}


def calculate_loss(logits, labels):
    """Calculate cross-entropy loss."""
    criterion = torch.nn.CrossEntropyLoss()
    shift_logits = logits[..., -1, :].contiguous()
    shift_labels = labels.contiguous().view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = criterion(shift_logits, shift_labels)
    return loss


def get_last_token(logits, attention_mask):
    """Get logits for the last token in each sequence."""
    last_token_indices = attention_mask.sum(1) - 1
    batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)