"""EpiDAG synthetic knowledge graph and task generator.

Generates dependency DAGs of depths d in {2, 4, 8, 16} with controlled:
  - Dependency depth
  - Branching factor
  - Distractor density
  - Grounded text sources
  - Deterministic arithmetic / comparison witnesses
  - Error injection for Error Propagation Coefficient (EPC_k)
"""

from __future__ import annotations

import random
from typing import Literal

from esc.benchmark.corpus import CorpusDocument, DocumentCorpus
from esc.benchmark.tasks import EpiDAGNode, EpiDAGTask
from esc.core.types import Evidence, StepSpec

COMPANY_NAMES = [
    "AcroDyn", "BioVect", "CipherSys", "DataForge", "EchoLabs", "FluxNet",
    "GeoMatrix", "HyperScale", "IonCore", "JunctionAI", "KineticScale", "LuminaCorp",
    "MicroVantage", "NovaSemis", "OmniLogic", "PentaVault", "QuantumRay", "RotorTech",
    "StrataWave", "TerraNova", "UltraFlux", "VertexAI", "WarpDrive", "XenoBio",
]

FOUNDERS = [
    "Elena Vance", "Marcus Chen", "Sarah Jenkins", "Devon Park", "Aaliyah Reed",
    "Vikram Patel", "Chloe Dupont", "Liam Gallagher", "Hana Tanaka", "Gabriel Ortiz",
]


def generate_synthetic_corpus(
    seed: int = 42,
    num_distractors: int = 8,
) -> tuple[dict[str, str], list[CorpusDocument]]:
    """Generate synthetic documents including true sources and distractor documents."""
    rng = random.Random(seed)
    docs: list[CorpusDocument] = []

    # Distractors
    for i in range(num_distractors):
        comp = f"DistractorCorp_{i+1}"
        rev = rng.randint(20, 900)
        year = rng.randint(1995, 2022)
        person = rng.choice(FOUNDERS)
        content = (
            f"{comp} was incorporated in {year} by {person}. "
            f"In the 2024 fiscal year, {comp} reported annual gross revenue of {rev} million USD. "
            f"The company maintains operations across several industrial sectors."
        )
        docs.append(
            CorpusDocument(
                doc_id=f"doc_distractor_{i+1}",
                title=f"Annual Report - {comp}",
                content=content,
                is_distractor=True,
            )
        )

    return {d.doc_id: d.content for d in docs}, docs


def generate_epidag_task(
    task_id: str,
    depth: int = 4,
    distractor_count: int = 6,
    seed: int = 42,
    inject_error_at_node: str | None = None,
) -> EpiDAGTask:
    """Generate an EpiDAG task with a specific dependency depth d in {2, 4, 8, 16}."""
    rng = random.Random(seed)

    alpha_comp = rng.choice(COMPANY_NAMES)
    founder = rng.choice(FOUNDERS)
    founding_year = rng.randint(2005, 2018)
    target_comp = f"Target_{rng.choice(COMPANY_NAMES)}"
    alpha_rev = rng.randint(100, 300)
    target_rev = rng.randint(400, 900)
    ratio_true = round(target_rev / alpha_rev, 4)
    threshold = round(ratio_true - 0.25, 2)
    threshold_holds = "true" if ratio_true > threshold else "false"

    docs: list[CorpusDocument] = []
    nodes: list[EpiDAGNode] = []
    ground_truth: dict[str, str] = {}

    # Document 1: Alpha profile
    doc_alpha_text = (
        f"{alpha_comp} is a specialized enterprise solutions provider. "
        f"The primary owner and principal founder of {alpha_comp} is {founder}. "
        f"For fiscal year 2024, {alpha_comp} registered an audited revenue of {alpha_rev} million USD."
    )
    docs.append(
        CorpusDocument(
            doc_id="doc_alpha_profile",
            title=f"Corporate Registry: {alpha_comp}",
            content=doc_alpha_text,
        )
    )

    # Document 2: Founder biography
    doc_founder_text = (
        f"{founder} launched their inaugural commercial venture in {founding_year}. "
        f"Later, {founder} oversaw the strategic acquisition of {target_comp} as an expansion arm."
    )
    docs.append(
        CorpusDocument(
            doc_id="doc_founder_bio",
            title=f"Executive Profile: {founder}",
            content=doc_founder_text,
        )
    )

    # Document 3: Target financial report
    doc_target_text = (
        f"{target_comp} announced fiscal year 2024 earnings today. "
        f"Consolidated top-line revenue reached {target_rev} million USD, reflecting strong year-over-year growth."
    )
    docs.append(
        CorpusDocument(
            doc_id="doc_target_earnings",
            title=f"Financial Disclosure: {target_comp}",
            content=doc_target_text,
        )
    )

    # Add distractors
    _, distractor_docs = generate_synthetic_corpus(seed=seed, num_distractors=distractor_count)
    docs.extend(distractor_docs)

    if depth == 2:
        # Depth 2:
        # N1: retrieve Alpha revenue (Type II grounded -> supported)
        # N2: evaluate if revenue exceeds 150 million (Type I deterministic -> verified)
        node_1 = EpiDAGNode(
            node_id="N1",
            step_spec=StepSpec(
                step_id="N1",
                goal=f"Retrieve the 2024 revenue of {alpha_comp} in million USD.",
                requires=[],
                permitted_sources=["doc_alpha_profile"],
                required_level="supported",
                expected_key="N1_alpha_revenue",
                witness_type="type_2",
            ),
            true_value=str(alpha_rev),
            true_evidence=[Evidence(source_id="doc_alpha_profile", span=f"revenue of {alpha_rev} million USD")],
            witness_type="type_2",
            level_depth=1,
        )
        t_holds = "true" if alpha_rev > 150 else "false"
        node_2 = EpiDAGNode(
            node_id="N2",
            step_spec=StepSpec(
                step_id="N2",
                goal=f"Determine whether {alpha_comp} 2024 revenue exceeds 150 million USD (return true or false).",
                requires=["N1_alpha_revenue"],
                permitted_sources=[],
                required_level="supported",
                expected_key="N2_final_decision",
                witness_type="type_1",
            ),
            true_value=t_holds,
            true_evidence=[],
            witness_type="type_1",
            level_depth=2,
        )
        nodes = [node_1, node_2]
        ground_truth = {"N1_alpha_revenue": str(alpha_rev), "N2_final_decision": t_holds}
        final_id = "N2_final_decision"
        question = f"Does {alpha_comp}'s 2024 revenue exceed 150 million USD?"

    elif depth == 4:
        # Depth 4:
        # N1: retrieve Alpha 2024 revenue (Type II grounded -> supported)
        # N2: identify principal founder of Alpha (Type II grounded -> supported)
        # N3: retrieve target 2024 revenue (Type II grounded -> supported)
        # N4: calculate ratio of target revenue to Alpha revenue (Type I arithmetic -> verified)
        node_1 = EpiDAGNode(
            node_id="N1",
            step_spec=StepSpec(
                step_id="N1",
                goal=f"Retrieve the 2024 revenue of {alpha_comp} in million USD.",
                requires=[],
                permitted_sources=["doc_alpha_profile"],
                required_level="supported",
                expected_key="N1_alpha_rev",
                witness_type="type_2",
            ),
            true_value=str(alpha_rev),
            true_evidence=[Evidence(source_id="doc_alpha_profile", span=f"revenue of {alpha_rev} million USD")],
            witness_type="type_2",
            level_depth=1,
        )
        node_2 = EpiDAGNode(
            node_id="N2",
            step_spec=StepSpec(
                step_id="N2",
                goal=f"Identify the principal owner/founder of {alpha_comp}.",
                requires=["N1_alpha_rev"],
                permitted_sources=["doc_alpha_profile"],
                required_level="supported",
                expected_key="N2_owner",
                witness_type="type_2",
            ),
            true_value=founder,
            true_evidence=[Evidence(source_id="doc_alpha_profile", span=f"principal founder of {alpha_comp} is {founder}")],
            witness_type="type_2",
            level_depth=2,
        )
        node_3 = EpiDAGNode(
            node_id="N3",
            step_spec=StepSpec(
                step_id="N3",
                goal=f"Retrieve the 2024 top-line revenue of the acquisition target {target_comp} overseen by {founder} in million USD.",
                requires=["N2_owner"],
                permitted_sources=["doc_founder_bio", "doc_target_earnings"],
                required_level="supported",
                expected_key="N3_target_rev",
                witness_type="type_2",
            ),
            true_value=str(target_rev),
            true_evidence=[Evidence(source_id="doc_target_earnings", span=f"revenue reached {target_rev} million USD")],
            witness_type="type_2",
            level_depth=3,
        )
        node_4 = EpiDAGNode(
            node_id="N4",
            step_spec=StepSpec(
                step_id="N4",
                goal=f"Calculate the ratio of target revenue ({target_rev}) to Alpha revenue ({alpha_rev}).",
                requires=["N3_target_rev", "N1_alpha_rev"],
                permitted_sources=[],
                required_level="supported",
                expected_key="N4_ratio",
                witness_type="type_1",
            ),
            true_value=str(ratio_true),
            true_evidence=[],
            witness_type="type_1",
            level_depth=4,
        )
        nodes = [node_1, node_2, node_3, node_4]
        ground_truth = {
            "N1_alpha_rev": str(alpha_rev),
            "N2_owner": founder,
            "N3_target_rev": str(target_rev),
            "N4_ratio": str(ratio_true),
        }
        final_id = "N4_ratio"
        question = (
            f"What is the ratio of 2024 revenue of the acquisition target owned by {alpha_comp}'s founder "
            f"to {alpha_comp}'s 2024 revenue?"
        )

    else:
        # Depths 8 and 16 (Full EpiDAG DAG matching the paper architecture)
        # N1: identify owner of Alpha
        # N2: identify founding year of owner's venture
        # N3: identify acquisition target of owner
        # N4: retrieve target 2024 revenue
        # N5: retrieve Alpha 2024 revenue
        # N6: calculate ratio (N4 / N5)
        # N7: determine whether threshold condition holds (N6 > threshold)
        # N8: final recommendation (approve if threshold holds, else deny)
        node_1 = EpiDAGNode(
            node_id="N1",
            step_spec=StepSpec(
                step_id="N1",
                goal=f"Identify the principal owner/founder of {alpha_comp}.",
                requires=[],
                permitted_sources=["doc_alpha_profile"],
                required_level="supported",
                expected_key="N1_owner",
                witness_type="type_2",
            ),
            true_value=founder,
            true_evidence=[Evidence(source_id="doc_alpha_profile", span=f"founder of {alpha_comp} is {founder}")],
            witness_type="type_2",
            level_depth=1,
        )
        node_2 = EpiDAGNode(
            node_id="N2",
            step_spec=StepSpec(
                step_id="N2",
                goal=f"Identify the founding year of {founder}'s inaugural venture.",
                requires=["N1_owner"],
                permitted_sources=["doc_founder_bio"],
                required_level="supported",
                expected_key="N2_founding_year",
                witness_type="type_2",
            ),
            true_value=str(founding_year),
            true_evidence=[Evidence(source_id="doc_founder_bio", span=f"inaugural commercial venture in {founding_year}")],
            witness_type="type_2",
            level_depth=2,
        )
        node_3 = EpiDAGNode(
            node_id="N3",
            step_spec=StepSpec(
                step_id="N3",
                goal=f"Identify the acquisition target enterprise overseen by {founder}.",
                requires=["N1_owner", "N2_founding_year"],
                permitted_sources=["doc_founder_bio"],
                required_level="supported",
                expected_key="N3_target",
                witness_type="type_2",
            ),
            true_value=target_comp,
            true_evidence=[Evidence(source_id="doc_founder_bio", span=f"acquisition of {target_comp}")],
            witness_type="type_2",
            level_depth=3,
        )
        node_4 = EpiDAGNode(
            node_id="N4",
            step_spec=StepSpec(
                step_id="N4",
                goal=f"Retrieve the 2024 revenue of target company in million USD.",
                requires=["N3_target"],
                permitted_sources=["doc_target_earnings"],
                required_level="supported",
                expected_key="N4_target_rev",
                witness_type="type_2",
            ),
            true_value=str(target_rev),
            true_evidence=[Evidence(source_id="doc_target_earnings", span=f"revenue reached {target_rev} million USD")],
            witness_type="type_2",
            level_depth=4,
        )
        node_5 = EpiDAGNode(
            node_id="N5",
            step_spec=StepSpec(
                step_id="N5",
                goal=f"Retrieve the 2024 revenue of {alpha_comp} in million USD.",
                requires=[],
                permitted_sources=["doc_alpha_profile"],
                required_level="supported",
                expected_key="N5_alpha_rev",
                witness_type="type_2",
            ),
            true_value=str(alpha_rev),
            true_evidence=[Evidence(source_id="doc_alpha_profile", span=f"revenue of {alpha_rev} million USD")],
            witness_type="type_2",
            level_depth=2,
        )
        node_6 = EpiDAGNode(
            node_id="N6",
            step_spec=StepSpec(
                step_id="N6",
                goal=f"Calculate the ratio of target revenue ({target_rev}) to Alpha revenue ({alpha_rev}).",
                requires=["N4_target_rev", "N5_alpha_rev"],
                permitted_sources=[],
                required_level="supported",
                expected_key="N6_ratio",
                witness_type="type_1",
            ),
            true_value=str(ratio_true),
            true_evidence=[],
            witness_type="type_1",
            level_depth=5,
        )
        node_7 = EpiDAGNode(
            node_id="N7",
            step_spec=StepSpec(
                step_id="N7",
                goal=f"Determine whether ratio exceeds threshold {threshold} (true or false).",
                requires=["N6_ratio"],
                permitted_sources=[],
                required_level="verified",
                expected_key="N7_threshold_check",
                witness_type="type_1",
            ),
            true_value=threshold_holds,
            true_evidence=[],
            witness_type="type_1",
            level_depth=6,
        )
        final_answer_val = "approved" if threshold_holds == "true" else "rejected"
        node_8 = EpiDAGNode(
            node_id="N8",
            step_spec=StepSpec(
                step_id="N8",
                goal="Generate final recommendation: approved if threshold passed, else rejected.",
                requires=["N7_threshold_check"],
                permitted_sources=[],
                required_level="verified",
                expected_key="N8_final_decision",
                witness_type="type_1",
            ),
            true_value=final_answer_val,
            true_evidence=[],
            witness_type="type_1",
            level_depth=7,
        )

        nodes = [node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8]
        ground_truth = {
            "N1_owner": founder,
            "N2_founding_year": str(founding_year),
            "N3_target": target_comp,
            "N4_target_rev": str(target_rev),
            "N5_alpha_rev": str(alpha_rev),
            "N6_ratio": str(ratio_true),
            "N7_threshold_check": threshold_holds,
            "N8_final_decision": final_answer_val,
        }
        final_id = "N8_final_decision"
        question = (
            f"Based on the enterprise hierarchy of {alpha_comp}, determine if the strategic acquisition "
            f"yields a revenue multiple above {threshold} and provide the final decision (approved/rejected)."
        )

        # If depth is 16, expand the DAG with 8 additional chained auditing / financial modeling nodes
        if depth >= 16:
            prev_decision = final_answer_val
            for step_idx in range(9, 17):
                n_id = f"N{step_idx}"
                key_id = f"{n_id}_audit_stage_{step_idx}"
                expected_val = f"stage_{step_idx}_cleared" if prev_decision == "approved" else f"stage_{step_idx}_halted"
                node = EpiDAGNode(
                    node_id=n_id,
                    step_spec=StepSpec(
                        step_id=n_id,
                        goal=f"Execute compliance check stage {step_idx} based on previous outcome.",
                        requires=[f"N{step_idx-1}_audit_stage_{step_idx-1}" if step_idx > 9 else "N8_final_decision"],
                        permitted_sources=[],
                        required_level="verified",
                        expected_key=key_id,
                        witness_type="type_1",
                    ),
                    true_value=expected_val,
                    true_evidence=[],
                    witness_type="type_1",
                    level_depth=step_idx,
                )
                nodes.append(node)
                ground_truth[key_id] = expected_val
            final_id = f"N16_audit_stage_16"
            question = f"{question} Complete all 16 compliance and risk verification stages."

    # Error injection for Error Propagation Coefficient (EPC_k)
    if inject_error_at_node:
        for n in nodes:
            if n.node_id == inject_error_at_node:
                n.injected_error_value = f"CORRUPT_{n.true_value}_WRONG"
                break

    corpus_map = {d.doc_id: d.content for d in docs}

    return EpiDAGTask(
        task_id=task_id,
        depth=depth,
        question=question,
        nodes=nodes,
        final_node_id=final_id,
        ground_truth_map=ground_truth,
        injected_error_node_id=inject_error_at_node,
        corpus_data=corpus_map,
        metadata={"alpha_comp": alpha_comp, "founder": founder, "seed": seed},
    )


def generate_task_suite(
    depths: list[int] = [2, 4, 8, 16],
    tasks_per_depth: int = 5,
    seed: int = 100,
    inject_errors: bool = False,
) -> list[EpiDAGTask]:
    """Generate a full suite of tasks across specified depths."""
    suite: list[EpiDAGTask] = []
    task_count = 0
    for d in depths:
        for i in range(tasks_per_depth):
            task_count += 1
            error_node = f"N{max(1, d // 2)}" if inject_errors else None
            task = generate_epidag_task(
                task_id=f"epidag_d{d}_t{i+1}",
                depth=d,
                distractor_count=5,
                seed=seed + task_count,
                inject_error_at_node=error_node,
            )
            suite.append(task)
    return suite
