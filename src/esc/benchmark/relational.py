"""Controlled relational-depth tasks with a fixed, equally accessible corpus.

Each relation is an independent random permutation over a fixed entity set.
Following d relations requires d dependent lookups. This measures relational
composition, not general semantic reasoning; Python may solve it deterministically.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Literal

from esc.benchmark.tasks import EpiDAGNode, EpiDAGTask
from esc.core.types import Evidence, LookupSpec, StepSpec

Split = Literal["train", "validation", "test", "dev"]


def generate_relational_suite(
    depths: list[int] | None = None,
    tasks_per_depth: int = 3,
    seed: int = 42,
    split: Split = "dev",
    width: int = 16,
    max_depth: int = 16,
) -> list[EpiDAGTask]:
    """Pair each world's start/corpus across depths; namespace worlds by split.

    width and max_depth determine corpus size, never the chosen prefix depth.
    Reordering or selecting depths does not change any world's generated data.
    """
    depths = [2, 4, 8, 16] if depths is None else depths
    if not depths or len(set(depths)) != len(depths) or any(d < 1 or d > max_depth for d in depths):
        raise ValueError("depths must be distinct positive prefixes no greater than max_depth")
    if split not in {"train", "validation", "test", "dev"}:
        raise ValueError("Unknown dataset split")
    if width < 2 or tasks_per_depth < 1 or max_depth < 1:
        raise ValueError("width must be >= 2, and task count and max_depth must be positive")

    suite = []
    for index in range(tasks_per_depth):
        digest = hashlib.sha256(f"relational_v2:{seed}:{split}:{index}".encode()).hexdigest()
        rng = random.Random(int(digest, 16))
        used = set()

        def opaque(prefix):
            while True:
                value = prefix + f"{rng.getrandbits(48):012x}"
                if value not in used:
                    used.add(value)
                    return value

        entities = [opaque("e") for _ in range(width)]
        relations = [opaque("r") for _ in range(max_depth)]
        start = rng.choice(entities)
        transitions = {}
        rows = []
        for relation in relations:
            targets = rng.sample(entities, len(entities))
            for subject, target in zip(entities, targets, strict=True):
                transitions[subject, relation] = target
                rows.append(json.dumps({"subject": subject, "relation": relation, "object": target}))

        rng.shuffle(rows)
        documents = []
        row_sources = {}
        # Shuffle groups and source names independently of the selected chain.
        for offset in range(0, len(rows), 8):
            source = opaque("s")
            group = rows[offset:offset + 8]
            documents.append((source, "\n".join(group)))
            for row in group:
                row_sources[row] = source
        rng.shuffle(documents)
        corpus = dict(documents)
        corpus_digest = hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest()
        allowed = list(corpus)
        chain = []
        subject = start
        for relation in relations:
            target = transitions[subject, relation]
            row = json.dumps({"subject": subject, "relation": relation, "object": target})
            chain.append((target, Evidence(source_id=row_sources[row], span=row)))
            subject = target

        for depth in depths:
            nodes = []
            for i in range(depth):
                parent = f"N{i}" if i else None
                lookup = LookupSpec(relation=relations[i], subject=start if i == 0 else None,
                                    subject_key=parent)
                subject_description = f"entity {start}" if i == 0 else f"the entity value of accepted fact {parent}"
                goal = (
                    f"In the authoritative JSON rows, find the row whose subject is {subject_description} "
                    f"and whose relation is {relations[i]}. Return only its object identifier. "
                    "Cite the complete JSON row and its document source ID. All documents are available; "
                    "select by subject and relation, not document position."
                )
                value, evidence = chain[i]
                key = f"N{i+1}"
                nodes.append(EpiDAGNode(
                    node_id=key, true_value=value, true_evidence=[evidence], level_depth=i+1,
                    witness_type="type_1",
                    step_spec=StepSpec(step_id=key, expected_key=key, goal=goal,
                                       requires=[parent] if parent else [], permitted_sources=allowed,
                                       required_level="verified", witness_type="type_1", lookup=lookup),
                ))
            question = (
                f"Start at entity {start}. Follow these relations in order: "
                + ", ".join(relations[:depth])
                + ". Each authoritative JSON row maps (subject, relation) to object; the object becomes "
                "the subject for the next relation. Return only the final entity identifier."
            )
            suite.append(EpiDAGTask(
                task_id=f"relational_v2_{split}_{digest[:16]}_d{depth}", depth=depth,
                question=question, nodes=nodes, final_node_id=f"N{depth}",
                ground_truth_map={node.node_id: node.true_value for node in nodes},
                corpus_data=corpus,
                metadata={"benchmark": "relational_v2", "split": split, "seed": seed,
                          "world_id": digest, "width": width, "max_depth": max_depth,
                          "corpus_sha256": corpus_digest, "source_policy": "full_corpus_every_step",
                          "witness_scope": "deterministic_authoritative_rows"},
            ))
    return suite
