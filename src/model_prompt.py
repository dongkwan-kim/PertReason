"""Generate the two published PertReasonQA three-way prompt presets."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data_samples"
PRESETS = ("cell_cond_path", "no_path")
PERT_TYPES = ("chemical", "geneticRNv3", "s_genetic2chemical", "s_genetic_combo")
TRIPLET_INSTRUCTION = '<triplet>A JSON list of causal edges from your reasoning, e.g., [[\\"A\\", \\"inhibits\\", \\"B\\"], [\\"B\\", \\"activates\\", \\"C\\"], [\\"C\\", \\"regulates\\", \\"D\\"]]. Entities must come from this set: ['


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Keep all input and output paths explicit and configurable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SAMPLE_DIR,
                        help="Directory containing input JSONL files.")
    parser.add_argument("--input-file", type=Path,
                        help="Generate prompts from this file instead of input-dir.")
    parser.add_argument("--thresholds-path", type=Path,
                        default=SAMPLE_DIR / "basal_thresholds.json",
                        help="JSON containing the published ks_thr and u_ks_thr maps.")
    parser.add_argument("--preset", choices=PRESETS, default="cell_cond_path")
    parser.add_argument("--pert-type", choices=PERT_TYPES,
                        help="Exact subtype override for records missing input.pert_type.")
    parser.add_argument("--output-dir", type=Path,
                        help="Write matching JSONL filenames here; otherwise print JSONL.")
    return parser.parse_args(argv)


def _normalize_output(record: dict[str, Any]) -> dict[str, Any]:
    """Accept public field names while giving existing legacy fields precedence."""
    output = dict(record.get("output") or {})
    for public, legacy in (
        ("triplets_ref", "triplets_cell_conditioned"),
        ("triplets_uncond", "triplets_unconditioned"),
        ("triplets_cell_cond", "triplets_all"),
    ):
        if legacy not in output and public in output:
            output[legacy] = output[public]
    return output


def _dataset(pert_type: str, cell_type: str) -> str:
    """Resolve only the dataset/cell combinations used by the public benchmark."""
    if pert_type == "chemical":
        if cell_type in ("a549", "k562", "mcf7"):
            return "Sciplex3"
        if cell_type in ("B cells", "Myeloid cells", "NK cells", "T cells"):
            return "OP3"
    elif pert_type == "geneticRNv3" and cell_type in ("hepg2", "jurkat", "k562", "rpe1"):
        return "RNv3"
    elif pert_type == "s_genetic2chemical" and cell_type == "hepg2":
        return "Demeo25"
    elif pert_type == "s_genetic_combo" and cell_type == "k562":
        return "Norman19"
    raise ValueError(f"Unsupported published perturbation/cell pair: {pert_type!r}, {cell_type!r}.")


def _perturbation_text(pert_type: str, perturbation: Any) -> tuple[str, str, str]:
    """Return the modality, question wording, and descriptive setting wording."""
    if pert_type in ("chemical", "s_genetic2chemical"):
        return "chemical", perturbation, perturbation
    if pert_type == "geneticRNv3":
        text = f"CRISPRi knockdown of {perturbation}"
        return "genetic", text, f"{text} (CRISPRi represses {perturbation})"
    if isinstance(perturbation, (list, tuple, set)):
        genes = " and ".join(str(gene).strip() for gene in perturbation if str(gene).strip())
    else:
        genes = str(perturbation).strip()
        if "+" in genes:
            genes = " and ".join(gene.strip() for gene in genes.split("+") if gene.strip())
    text = f"CRISPRa overexpression of {genes}"
    return "genetic", text, f"{text} (CRISPRa activates {genes})"


def _add_entities(entities: set[str], value: Any) -> None:
    if isinstance(value, str):
        entities.add(value)
    elif isinstance(value, list):
        entities.update(entity for entity in value if isinstance(entity, str))


def _candidate_entities(triplets: list) -> set[str]:
    entities: set[str] = set()
    for triplet in triplets:
        if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
            _add_entities(entities, triplet[0])
            _add_entities(entities, triplet[2])
    return entities


def _basal_lines(record: dict, output: dict, thresholds: dict, dataset: str) -> list[str]:
    basal = record.get("basal_context") or {}
    if not basal:
        return []
    input_data = record["input"]
    candidates = output.get("triplets_all")
    if not output:
        genes = set(basal)
    elif isinstance(candidates, list):
        genes = _candidate_entities(candidates)
    else:
        genes: set[str] = set()
        _add_entities(genes, input_data.get("pert"))
        _add_entities(genes, input_data.get("gene"))
    cell = input_data["cell_type"]
    try:
        low = thresholds["ks_thr"][dataset][cell]
        high = thresholds["u_ks_thr"][dataset][cell]
    except KeyError as error:
        raise ValueError(f"Missing published basal thresholds for {dataset}/{cell}.") from error
    lines = ["\n## Basal Expression Context", "The basal gene expression levels in the pathway are:"]
    for gene, value in sorted(basal.items(), key=lambda item: -item[1]):
        if gene in genes:
            state = "low" if value <= low else "high" if value >= high else "medium"
            lines.append(f"- {gene}: {state}")
    return lines


def _path_text(triplets: list) -> str | None:
    """Preserve the published flat BFS traversal, including duplicate edges."""
    if not triplets:
        return None
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    nodes: set[str] = set()
    targets: set[str] = set()
    for source, relation, target in triplets:
        adjacency[source].append((relation, target))
        nodes.update((source, target))
        targets.add(target)
    roots = sorted(nodes - targets) or [triplets[0][0]]
    pending = deque(roots)
    visited: set[str] = set()
    lines = []
    while pending:
        node = pending.popleft()
        if node in visited:
            continue
        visited.add(node)
        for relation, target in sorted(adjacency.get(node, [])):
            lines.append(f"- {node} {relation} {target}")
        for target in sorted(target for _, target in adjacency.get(node, [])):
            if target not in visited and target not in pending:
                pending.append(target)
    return "\n".join(lines) if lines else "No valid pathway structure found."


def _entity_list(record: dict, output: dict) -> str:
    candidates = output.get("triplets_all")
    genes = _candidate_entities(candidates) if isinstance(candidates, list) else set()
    basal_genes = [gene for gene in (record.get("basal_context") or {}) if gene in genes]
    perturbation = record["input"]["pert"]
    if isinstance(perturbation, str):
        entities = [perturbation] + basal_genes
    elif isinstance(perturbation, list):
        entities = perturbation + basal_genes
    else:
        raise ValueError(f"Unexpected perturbation format: {perturbation!r}.")
    return ", ".join(f'"{entity}"' for entity in dict.fromkeys(entities))


def format_prompt(
    record: dict[str, Any],
    thresholds: dict[str, Any],
    preset: str = "cell_cond_path",
    pert_type: str | None = None,
) -> str:
    """Format a public reasoning/outcome record without modifying its contents.

    Exact perturbation subtypes belong in input.pert_type. Already formatted
    releases may retain only a generic top-level modality; supply pert_type
    explicitly in that case to select the correct thresholds and CRISPR wording.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {PRESETS}.")
    input_data = record["input"]
    if input_data.get("task") != "3way":
        raise ValueError("The published presets require input.task='3way'.")
    subtype = pert_type if pert_type is not None else input_data.get("pert_type")
    if subtype not in PERT_TYPES:
        raise ValueError("Set input.pert_type or --pert-type to an exact published subtype; generic modalities cannot identify the dataset.")
    cell = input_data["cell_type"]
    dataset = _dataset(subtype, cell)
    modality, perturbation, setting = _perturbation_text(subtype, input_data["pert"])
    gene = input_data["gene"]
    output = _normalize_output(record)
    context_lines = _basal_lines(record, output, thresholds, dataset)
    goal = f"Your goal is to determine: Is the expression of {gene} up, down, or unchanged after {modality} perturbation {perturbation} in {cell}?"
    if preset == "cell_cond_path":
        goal += "\nAnswer the question based on the context and the retrieved knowledges. Note that the retrieved knowledges may contain irrelevant information. Focus only on pathways that causally connect the perturbation to the effect gene."
        triplets = output.get("triplets_all")
        paths = _path_text(triplets) if isinstance(triplets, list) else None
        if paths:
            context_lines.extend(("\n## Retrieved Knowledges", paths))
    else:
        goal += "\nAnswer the question based on the context. You must reason using your own knowledge on the biological pathway that connects the perturbation to the effect gene."
    context = "\n".join(context_lines)
    question = f'Is the expression of {gene} up, down, or unchanged after the {modality} perturbation {perturbation} in {cell}? Answer with either "up", "down", or "unchanged".'
    instructions = (
        "Please reason step by step. Structure your response in exactly 3 blocks:\n"
        "1. <thinking>Your step-by-step biological reasoning here.</thinking>\n"
        "2. <answer>up</answer> OR <answer>down</answer> OR <answer>unchanged</answer>\n"
        f"3. {TRIPLET_INSTRUCTION}{_entity_list(record, output)}].</triplet>"
    )
    return (
        f"You are an expert biologist. {goal}\n\n"
        "## Task Information\n"
        f"- **Cell Type:** {cell}\n"
        f"- **Perturbation ({modality}):** {setting}\n"
        f"- **Effect Gene:** {gene}\n"
        f"{context}\n\n"
        "## Question\n"
        f"{question}\n{instructions}"
    )


def format_jsonl(
    input_path: str | Path,
    thresholds: dict[str, Any],
    preset: str = "cell_cond_path",
    pert_type: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield original JSONL records with their prompt field regenerated."""
    input_path = Path(input_path)
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prompt = format_prompt(record, thresholds, preset, pert_type)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error
            yield {**record, "prompt": prompt}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sample_input = args.input_dir / "samples_input.jsonl"
    if args.input_file is not None:
        files = [args.input_file]
    else:
        files = sorted(path for path in args.input_dir.glob("*.jsonl")
                       if not (sample_input.is_file() and path.name == "samples_output.jsonl"))
    if not files:
        raise ValueError(f"No JSONL files found in {args.input_dir}.")
    with args.thresholds_path.open(encoding="utf-8") as source:
        thresholds = json.load(source)
    if args.output_dir:
        if any((args.output_dir / path.name).resolve() == path.resolve() for path in files):
            raise ValueError("Output files must differ from input files.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        records = format_jsonl(path, thresholds, args.preset, args.pert_type)
        if args.output_dir:
            temporary = None
            try:
                with NamedTemporaryFile("w", encoding="utf-8", dir=args.output_dir,
                                        delete=False) as destination:
                    temporary = Path(destination.name)
                    for record in records:
                        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                if (args.output_dir / path.name).exists():
                    temporary.chmod((args.output_dir / path.name).stat().st_mode & 0o777)
                temporary.replace(args.output_dir / path.name)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        else:
            for record in records:
                print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
