"""Evaluate structured causal triplets without an external knowledge graph.

The module retains legacy extraction, single-start connectivity, and no-KG
metrics, with optional GO functional similarity from a local MSigDB GMT file.
Records are paired by ID. Reference paths
may use either the legacy ``triplets_cell_conditioned`` field or the released ``triplets_ref`` field.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "data_samples"
# GOSim uses MSigDB Human C5 GO BP v2023.2, gene-symbol GMT:
# c5.go.bp.v2023.2.Hs.symbols.gmt is bundled in data_aux/; override with --go-gmt.
DEFAULT_GO_GMT = DEFAULT_INPUT_DIR.parent / "data_aux" / "c5.go.bp.v2023.2.Hs.symbols.gmt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse all filesystem paths before loading any records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing ground-truth JSONL files.")
    parser.add_argument("--input-file", type=Path,
                        help="Evaluate this ground-truth file instead of input-dir.")
    predictions = parser.add_mutually_exclusive_group()
    predictions.add_argument("--prediction-dir", type=Path,
                             help="Directory of prediction files with matching input filenames.")
    predictions.add_argument("--prediction-file", type=Path,
                             help="Prediction file paired with an explicit input-file.")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional directory for per-file JSON reports. Reports are always printed to stdout.")
    parser.add_argument("--go-gmt", type=Path,
                        help="MSigDB GO BP symbols GMT. Defaults to data_aux/c5.go.bp.v2023.2.Hs.symbols.gmt when present; otherwise GOSim is unavailable.")
    args = parser.parse_args(argv)
    if args.prediction_file is not None and args.input_file is None:
        parser.error("--prediction-file requires --input-file.")
    return args


@dataclass(frozen=True)
class CausalEdge:
    """A single directed edge in the biological causal graph."""
    source: str
    target: str
    sign: str  # "activates", "inhibits", "regulates"

    @property
    def numeric(self) -> Optional[int]:
        return {"activates": +1, "inhibits": -1}.get(self.sign)

    def __repr__(self) -> str:
        sym = {"activates": "─(+)→", "inhibits": "─(-)→", "regulates": "─(?)→"}
        return f"{self.source} {sym.get(self.sign, '→')} {self.target}"


@dataclass
class PromptData:
    """Structured content parsed from the prompt text."""
    cell_type: str
    pert_type: str          # "chemical" | "genetic"
    perturbation: str       # drug name or perturbation description
    pert_gene: str          # actual gene name for genetic perturbation
    effect_gene: str
    task: str               # always "3way"
    basal_expression: Dict[str, str] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    all_genes: Set[str] = field(default_factory=set)


@dataclass
class SampleRecord:
    """
    One evaluation sample, combining GT data (from input file)
    and model prediction (from output file).

    Fields
    ------
    id : sample identifier (row index if the ground-truth record has no ID)
    prompt : full prompt text
    model_output : raw model generation (contains <thinking>, <answer>, <triplet>)
    gt_label : ground-truth answer ("up", "down", "unchanged")
    gt_triplets : GT causal path from triplets_cell_conditioned
    gt_reasoning : GT reasoning text (if available)
    prompt_data : structured parsed metadata from prompt
    basal_context : per-gene basal expression values
    pert_type : "chemical" or "genetic"
    """
    id: Any
    prompt: str
    model_output: str
    gt_label: str
    gt_triplets: List[Tuple[str, str, str]] = field(default_factory=list)
    gt_reasoning: str = ""
    prompt_data: Optional[PromptData] = None
    basal_context: Dict[str, Any] = field(default_factory=dict)
    pert_type: str = ""


def extract_answer(text: str) -> Optional[str]:
    """Extract and normalize the answer from <answer>...</answer> block.

    When models self-correct, they may emit multiple <answer> blocks.
    We use the **last** valid <answer> block, which is the corrected one.
    """
    valid = {"up", "down", "unchanged"}

    # Primary: LAST <answer>...</answer>
    matches = re.findall(r"<answer>\s*(\w+)\s*</answer>", text, re.IGNORECASE)
    for ans_raw in reversed(matches):
        ans = ans_raw.strip().lower()
        if ans in valid:
            return ans

    # Fallback: \boxed{...}
    m = re.search(r"\\boxed\{(\w+)\}", text)
    if m:
        ans = m.group(1).strip().lower()
        if ans in valid:
            return ans

    # Fallback: last word
    text_clean = text.strip()
    if text_clean:
        last_word = text_clean.rstrip(".").split()[-1].lower()
        if last_word in valid:
            return last_word

    return None


def extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract causal triplets from <triplet>...</triplet> block.

    When models self-correct, they may emit multiple <triplet> blocks.
    We use the **last** block, which is the corrected one.

    Handles both proper JSON arrays and common formatting variations
    (missing outer brackets, trailing commas, etc.).
    """
    blocks = re.findall(r"<triplet>(.*?)</triplet>", text, re.DOTALL)
    if not blocks:
        return []

    # Try each block from last to first until one parses successfully
    for raw in reversed(blocks):
        raw = raw.strip()
        if not raw:
            continue

        result = _try_parse_triplet_block(raw)
        if result:
            return result

    return []


def _try_parse_triplet_block(raw: str) -> List[Tuple[str, str, str]]:
    """Attempt to parse a single raw triplet block string."""
    # Try parsing as proper JSON array of arrays
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = _validate_triplets(parsed)
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Common issue: list of lists without outer brackets
    # e.g. ["A", "inhibits", "B"],\n["C", "activates", "D"]
    if not raw.startswith("[["):
        wrapped = "[" + raw.rstrip(",").rstrip() + "]"
        try:
            parsed = json.loads(wrapped)
            if isinstance(parsed, list):
                result = _validate_triplets(parsed)
                if result:
                    return result
        except json.JSONDecodeError:
            pass

    # Regex fallback: extract individual triplets
    triplets = []
    for match in re.finditer(
        r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', raw
    ):
        triplets.append((match.group(1), _normalize_relation(match.group(2)), match.group(3)))
    return [t for t in triplets if t[1] is not None]


_RELATION_MAP = {
    # Standard (identity)
    "activates": "activates",
    "activate": "activates",
    "inhibits": "inhibits",
    "regulates": "regulates",
    "regulate": "regulates",
    # Positive / activating synonyms
    "upregulates": "activates",
    "increases": "activates",
    "promotes": "activates",
    "induces": "activates",
    "stabilizes": "activates",
    "supports": "activates",
    "positively regulates": "activates",
    "indirectly activates": "activates",
    "induces upregulation of": "activates",
    "phosphorylates": "activates",
    "is upregulated": "activates",
    # Negative / inhibiting synonyms
    "suppresses": "inhibits",
    "downregulates": "inhibits",
    "reduces": "inhibits",
    "represses": "inhibits",
    "inactivates": "inhibits",
    "negatively regulates": "inhibits",
    "indirectly suppresses": "inhibits",
    "indirectly reduces": "inhibits",
    "phosphorylates and destabilizes": "inhibits",
    "inhibited by": "inhibits",
    "inhibited_by": "inhibits",
    # Ambiguous → regulates
    "modulates": "regulates",
    "influences": "regulates",
    "affects": "regulates",
    "indirectly regulates": "regulates",
    "leads to": "regulates",
    "involved in": "regulates",
    "responds to": "regulates",
    "regulated by": "regulates",
    "regulated_by": "regulates",
    "is regulated by": "regulates",
    "upregulated by": "regulates",
    "downregulated by": "regulates",
    "repressed by": "regulates",
    "is stabilized by": "regulates",
    "is not regulated by": None,   # negation → drop
    "not regulated": None,         # negation → drop
    "does not regulate": None,     # negation → drop
    "no regulatory link": None,    # negation → drop
    "expression": "regulates",
}


def _normalize_relation(rel: str) -> Optional[str]:
    """Map a relation string to canonical form, or None if unrecognized."""
    return _RELATION_MAP.get(rel.lower().strip())


def _validate_triplets(parsed: list) -> List[Tuple[str, str, str]]:
    """Validate and convert parsed JSON to list of triplet tuples.

    Non-standard relations (e.g. 'suppresses', 'upregulates') are
    normalized to the canonical set {activates, inhibits, regulates}.
    Triplets with truly unknown relations are discarded.
    """
    result = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            src, tgt = str(item[0]), str(item[2])
            norm_rel = _normalize_relation(str(item[1]))
            if norm_rel is not None:
                result.append((src, norm_rel, tgt))
    return result


def parse_prompt(prompt: str) -> PromptData:
    """Extract structured info from a PerturbQA prompt string."""
    cell = re.search(r"\*\*Cell Type:\*\*\s*(.+)", prompt)
    pert = re.search(r"\*\*Perturbation \((\w+)\):\*\*\s*(.+)", prompt)
    gene = re.search(r"\*\*Effect Gene:\*\*\s*(.+)", prompt)

    # Basal expression
    basal: Dict[str, str] = {}
    basal_sec = re.search(
        r"basal gene expression.*?:\n(.*?)(?=\n##)", prompt, re.DOTALL | re.IGNORECASE
    )
    if basal_sec:
        for m in re.finditer(r"-\s+(\S+):\s*(\w+)", basal_sec.group(1)):
            basal[m.group(1)] = m.group(2)

    # Knowledge edges
    edges: List[CausalEdge] = []
    know_sec = re.search(r"## Retrieved Knowledges?\n(.*?)(?=\n##)", prompt, re.DOTALL)
    if know_sec:
        for line in know_sec.group(1).split("\n"):
            line = line.strip().lstrip("- ")
            if line:
                e = _parse_edge_text(line)
                if e:
                    edges.append(e)

    all_genes: Set[str] = set()
    for e in edges:
        all_genes.add(e.source)
        all_genes.add(e.target)
    all_genes.update(basal.keys())

    pert_name = pert.group(2).strip() if pert else ""
    pert_type_str = pert.group(1).strip() if pert else ""
    effect_name = gene.group(1).strip() if gene else ""

    # For genetic perturbations, extract gene name
    pert_gene = ""
    if pert_type_str.lower() == "genetic":
        gene_match = re.search(
            r"(?:knockdown|knockout|overexpression|activation)\s+(?:of\s+)?(\S+)",
            pert_name, re.IGNORECASE,
        )
        if gene_match:
            pert_gene = gene_match.group(1)
        else:
            parts = pert_name.split()
            if parts:
                pert_gene = parts[-1]
    else:
        pert_gene = pert_name

    if pert_name:
        all_genes.add(pert_name)
    if pert_gene and pert_gene != pert_name:
        all_genes.add(pert_gene)
    if effect_name:
        all_genes.add(effect_name)

    return PromptData(
        cell_type=cell.group(1).strip() if cell else "",
        pert_type=pert_type_str,
        perturbation=pert_name,
        pert_gene=pert_gene,
        effect_gene=effect_name,
        task="3way",
        basal_expression=basal,
        edges=edges,
        all_genes=all_genes,
    )


def _parse_edge_text(text: str) -> Optional[CausalEdge]:
    """Parse 'A inhibits B' → CausalEdge."""
    for pat, sign in [
        (r"(.+?)\s+inhibits?\s+(.+)", "inhibits"),
        (r"(.+?)\s+activates?\s+(.+)", "activates"),
        (r"(.+?)\s+regulates?\s+(.+)", "regulates"),
    ]:
        m = re.match(pat, text.strip(), re.IGNORECASE)
        if m:
            return CausalEdge(m.group(1).strip(), m.group(2).strip(), sign)
    return None


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning a list of dicts."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@dataclass
class SampleResult:
    """Answer and KG-independent reasoning results for one sample."""

    id: Any
    cell_type: str = ""
    pert_type: str = ""
    perturbation: str = ""
    effect_gene: str = ""
    task: str = ""
    gt_answer: str = ""
    model_answer: str = ""
    answer_correct: Optional[bool] = None
    answer_parse_fail: bool = False
    edge_f1_strict: float = 0.0
    edge_f1_relaxed: float = 0.0
    edge_recall_strict: float = 0.0
    edge_precision_strict: float = 0.0
    path_connectivity: str = ""
    num_model_triplets: int = 0
    go_sim_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_edge_f1(
    gt_triplets: List[Tuple[str, str, str]],
    model_triplets: List[Tuple[str, str, str]],
    strict: bool = True,
) -> Dict[str, float]:
    """
    Compare GT path triplets with model-extracted triplets.

    strict=True  → (src, rel, tgt) must match exactly
    strict=False → only (src, tgt) must match (topology-only)
    """
    if strict:
        gt_set = {(s, r, t) for s, r, t in gt_triplets}
        model_set = {(s, r, t) for s, r, t in model_triplets}
    else:
        gt_set = {(s, t) for s, _, t in gt_triplets}
        model_set = {(s, t) for s, _, t in model_triplets}

    if not gt_set and not model_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not model_set:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    tp = len(gt_set & model_set)
    precision = tp / len(model_set)
    recall = tp / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def check_path_connectivity(
    model_triplets: List[Tuple[str, str, str]],
    start_node: str,
    end_node: str,
) -> str:
    """
    Check if model triplets form a connected directed path
    from start_node (perturbation) to end_node (effect gene).

    Returns "connected", "disconnected", or "empty".
    """
    assert start_node and start_node.strip(), "Path connectivity requires a non-empty perturbation start node."
    assert end_node and end_node.strip(), "Path connectivity requires a non-empty effect-gene end node."

    if not model_triplets:
        return "empty"

    adj: Dict[str, Set[str]] = defaultdict(set)
    for src, _, tgt in model_triplets:
        adj[src].add(tgt)

    # BFS from start_node
    visited: Set[str] = set()
    queue = [start_node]
    visited.add(start_node)

    while queue:
        curr = queue.pop(0)
        if curr == end_node:
            return "connected"
        for neighbor in adj.get(curr, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return "disconnected"


def load_paired_samples(gt_path: Path | str, pred_path: Path | str) -> List[SampleRecord]:
    """Pair records by ID, preserving ground-truth order.

    Every prediction must provide an ID. For ID-based matching, both files must
    provide unique, non-null IDs for every record.
    Unmatched IDs are skipped with a warning rather than paired by position.

    The ground-truth prompt takes precedence over a prediction prompt. Endpoints
    come from the original canonical prompt parser, including its first-root
    interpretation of genetic combination descriptions. This function does not
    replace that behavior with structured-input or multi-root inference.
    """
    gt_records = load_jsonl(gt_path)
    pred_records = load_jsonl(pred_path)
    if any("id" not in record for record in pred_records):
        raise ValueError("Every prediction record must have an ID for ID-based matching.")
    if any("id" in record for record in pred_records):
        def index_records(records, name):
            indexed = {}
            for record in records:
                if "id" not in record or record["id"] is None:
                    raise ValueError(f"Every {name} record must have an ID for ID-based matching.")
                record_id = record["id"]
                if record_id in indexed:
                    raise ValueError(f"Duplicate {name} ID: {record_id!r}.")
                indexed[record_id] = record
            return indexed

        gt_by_id = index_records(gt_records, "ground-truth")
        pred_by_id = index_records(pred_records, "prediction")
        missing_ids = gt_by_id.keys() - pred_by_id.keys()
        unknown_ids = pred_by_id.keys() - gt_by_id.keys()
        if missing_ids or unknown_ids:
            print(
                f"WARNING: {len(missing_ids)} ground-truth IDs have no prediction; "
                f"{len(unknown_ids)} prediction IDs are not in ground truth. "
                "Evaluating matching IDs only.",
                file=sys.stderr,
            )
        pairs = [(gt, pred_by_id[gt["id"]]) for gt in gt_records if gt["id"] in pred_by_id]
    else:
        if len(gt_records) != len(pred_records):
            print(
                f"WARNING: GT has {len(gt_records)} samples, predictions has "
                f"{len(pred_records)} samples. Using min({len(gt_records)}, {len(pred_records)}).",
                file=sys.stderr,
            )
        pairs = list(zip(gt_records, pred_records))

    samples = []
    for index, (gt, pred) in enumerate(pairs):
        prompt = gt.get("prompt", "") or pred.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f"Record {gt.get('id', index)!r} requires a non-empty prompt "
                "in the ground-truth or prediction file."
            )
        output = gt.get("output", {}) or {}
        triplets = output.get("triplets_cell_conditioned", output.get("triplets_ref", []))
        label = gt.get("label", gt.get("input", {}).get("label", ""))
        samples.append(SampleRecord(
            id=gt.get("id", index),
            prompt=prompt,
            model_output=pred.get("model_output", ""),
            gt_label=label,
            gt_triplets=[tuple(triplet) for triplet in triplets if len(triplet) == 3],
            prompt_data=parse_prompt(prompt),
            basal_context=gt.get("basal_context", {}) or {},
            pert_type=gt.get("pert_type", ""),
        ))
    return samples


def evaluate_symbolic(samples: List[SampleRecord], results: List[SampleResult]) -> Dict[str, Any]:
    """Populate sample results and summarize the original no-KG metrics.

    A missing extracted model path receives zero edge scores at this layer,
    including when the reference path is empty. This deliberately differs from
    the low-level empty-set behavior of compute_edge_f1(). Missing prompt_data
    records are skipped but retained in the denominator, as in the original.
    """
    strict_f1s, relaxed_f1s = [], []
    strict_recalls, relaxed_recalls, strict_precisions = [], [], []
    connectivity_counts = {"connected": 0, "disconnected": 0, "empty": 0}
    total = len(samples)

    for sample, result in zip(samples, results):
        prompt_data = sample.prompt_data
        if not prompt_data:
            continue

        model_triplets = extract_triplets(sample.model_output)
        result.num_model_triplets = len(model_triplets)
        if not model_triplets:
            result.edge_f1_strict = 0.0
            result.edge_f1_relaxed = 0.0
            result.edge_recall_strict = 0.0
            result.edge_precision_strict = 0.0
            result.path_connectivity = "empty"
            strict = relaxed = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            strict = compute_edge_f1(list(sample.gt_triplets), model_triplets, strict=True)
            relaxed = compute_edge_f1(list(sample.gt_triplets), model_triplets, strict=False)
            result.edge_f1_strict = strict["f1"]
            result.edge_f1_relaxed = relaxed["f1"]
            result.edge_recall_strict = strict["recall"]
            result.edge_precision_strict = strict["precision"]
            start = prompt_data.pert_gene if prompt_data.pert_gene else prompt_data.perturbation
            result.path_connectivity = check_path_connectivity(model_triplets, start, prompt_data.effect_gene)

        strict_f1s.append(strict["f1"])
        relaxed_f1s.append(relaxed["f1"])
        strict_recalls.append(strict["recall"])
        relaxed_recalls.append(relaxed["recall"])
        strict_precisions.append(strict["precision"])
        connectivity_counts[result.path_connectivity] += 1

    if total == 0:
        return {}

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "edge_f1_strict_mean": mean(strict_f1s),
        "edge_f1_relaxed_mean": mean(relaxed_f1s),
        "edge_recall_strict_mean": mean(strict_recalls),
        "edge_recall_relaxed_mean": mean(relaxed_recalls),
        "edge_precision_strict_mean": mean(strict_precisions),
        "path_connected_rate": connectivity_counts["connected"] / total,
        "path_disconnected_rate": connectivity_counts["disconnected"] / total,
        "path_empty_rate": connectivity_counts["empty"] / total,
        "num_samples": total,
    }


class GoAnnotations:
    """Original IDF-weighted GO profiles loaded from local MSigDB GMT files."""

    def __init__(self):
        self.gene_to_terms: Dict[str, Set[str]] = defaultdict(set)
        self._term_to_genes: Dict[str, Set[str]] = defaultdict(set)
        self.idf: Dict[str, float] = {}
        self._finalized = False

    def load_gmt(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GMT file not found: {path}")
        self._finalized = False
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                term_name = parts[0]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                for gene in genes:
                    self.gene_to_terms[gene].add(term_name)
                    self._term_to_genes[term_name].add(gene)

    def finalize(self) -> None:
        """IDF(term) = log(total annotated genes / genes annotated to term)."""
        n_total = len(self.gene_to_terms)
        if n_total == 0:
            raise RuntimeError("No GMT data loaded.  Call load_gmt() first.")
        self.idf = {}
        for term, genes in self._term_to_genes.items():
            n_term = len(genes)
            self.idf[term] = math.log(n_total / n_term) if n_term > 0 else 0.0
        self._finalized = True

    def gene_set_vector(self, genes: Set[str] | List[str]) -> Dict[str, float]:
        """Union of GO terms with max-IDF aggregation, as in the original."""
        if not self._finalized:
            raise RuntimeError("Call finalize() before scoring.")
        vec: Dict[str, float] = {}
        for gene in genes:
            for term in self.gene_to_terms.get(gene, set()):
                w = self.idf.get(term, 0.0)
                if w > vec.get(term, 0.0):
                    vec[term] = w
        return vec

    @staticmethod
    def cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(v1[t] * v2[t] for t in v1 if t in v2)
        norm1 = math.sqrt(sum(x * x for x in v1.values()))
        norm2 = math.sqrt(sum(x * x for x in v2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)


def evaluate_go_similarity(
    samples: List[SampleRecord],
    results: List[SampleResult],
    go_annotations: Optional[GoAnnotations] = None,
) -> Dict[str, Any]:
    """Match the original GO scoring, anchor exclusion, and empty-path rules."""
    if go_annotations is None:
        return {"go_sim_available": False, "go_sim_mean": None, "go_sim_median": None}

    anno = go_annotations
    num_no_gt_triplets = 0
    num_no_go_coverage = 0
    scores: List[float] = []
    for sample, result in zip(samples, results):
        gt_triplets = sample.gt_triplets
        if not gt_triplets:
            num_no_gt_triplets += 1
            result.go_sim_score = None
            continue
        gt_genes_full: Set[str] = set()
        for src, _sign, tgt in gt_triplets:
            gt_genes_full.add(src)
            gt_genes_full.add(tgt)
        pred_triplets = extract_triplets(sample.model_output)
        pred_genes_full: Set[str] = set()
        for src, _sign, tgt in pred_triplets:
            pred_genes_full.add(src)
            pred_genes_full.add(tgt)

        # Score intermediate nodes; fall back to full genes for short chains.
        anchor_genes: Set[str] = set()
        if sample.prompt_data:
            if sample.prompt_data.pert_gene:
                anchor_genes.add(sample.prompt_data.pert_gene)
            if sample.prompt_data.effect_gene:
                anchor_genes.add(sample.prompt_data.effect_gene)
        gt_genes = gt_genes_full - anchor_genes
        pred_genes = pred_genes_full - anchor_genes
        if not gt_genes:
            gt_genes = gt_genes_full
        if not pred_genes:
            pred_genes = pred_genes_full
        v_gt = anno.gene_set_vector(gt_genes)
        v_pred = anno.gene_set_vector(pred_genes) if pred_genes else {}
        if not v_gt:
            num_no_go_coverage += 1
            result.go_sim_score = 0.0
            scores.append(0.0)
            continue
        sc = GoAnnotations.cosine(v_gt, v_pred)
        result.go_sim_score = sc
        scores.append(sc)

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mid = n // 2
    return {
        "go_sim_available": True,
        "go_sim_mean": sum(sorted_scores) / n if n else None,
        "go_sim_median": (sorted_scores[mid] if n % 2 else
                          (sorted_scores[mid - 1] + sorted_scores[mid]) / 2) if n else None,
        "go_sim_min": sorted_scores[0] if n else None,
        "go_sim_max": sorted_scores[-1] if n else None,
        "num_scored": n,
        "num_no_gt_triplets": num_no_gt_triplets,
        "num_no_go_coverage": num_no_go_coverage,
    }


def evaluate_file(
    gt_path: Path | str, pred_path: Path | str | None = None,
    go_annotations: Optional[GoAnnotations] = None,
) -> Dict[str, Any]:
    """Evaluate one pair of JSONL files, or inline model_output demo records."""
    samples = load_paired_samples(gt_path, pred_path if pred_path is not None else gt_path)
    results = []
    for sample in samples:
        answer = extract_answer(sample.model_output)
        prompt_data = sample.prompt_data
        results.append(SampleResult(
            id=sample.id,
            cell_type=prompt_data.cell_type if prompt_data else "",
            pert_type=sample.pert_type or (prompt_data.pert_type if prompt_data else ""),
            perturbation=prompt_data.perturbation if prompt_data else "",
            effect_gene=prompt_data.effect_gene if prompt_data else "",
            task=prompt_data.task if prompt_data else "",
            gt_answer=sample.gt_label,
            model_answer=answer or "",
            answer_correct=answer == sample.gt_label,
            answer_parse_fail=answer is None,
        ))
    summary = evaluate_symbolic(samples, results)
    if samples:
        summary.update(evaluate_go_similarity(samples, results, go_annotations))
    return {
        "file_name": Path(gt_path).name,
        "summary": summary,
        "sample_results": [result.to_dict() for result in results],
    }




def main(argv: list[str] | None = None) -> None:
    """Evaluate explicit files, the bundled real pair, or same-named directory files."""
    args = parse_args(argv)
    sample_input = args.input_dir / "samples_input.jsonl"
    sample_output = args.input_dir / "samples_output.jsonl"
    if args.input_file is not None:
        prediction = args.prediction_file
        if prediction is None and args.prediction_dir is not None:
            prediction = args.prediction_dir / args.input_file.name
        pairs = [(args.input_file, prediction)]
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {args.input_dir}")
        files = sorted(args.input_dir.glob("*.jsonl"))
        if sample_input in files:
            files = [path for path in files if path != sample_output]
        if not files:
            raise SystemExit(f"No JSONL files found in: {args.input_dir}")
        pairs = [
            (path, args.prediction_dir / path.name if args.prediction_dir is not None
             else sample_output if path == sample_input and sample_output.is_file() else None)
            for path in files
        ]
    go_gmt = args.go_gmt
    if go_gmt is None and DEFAULT_GO_GMT.is_file():
        go_gmt = DEFAULT_GO_GMT
    go_annotations = None
    if go_gmt is not None:
        go_annotations = GoAnnotations()
        go_annotations.load_gmt(go_gmt)
        go_annotations.finalize()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for gt_path, pred_path in pairs:
        report = evaluate_file(gt_path, pred_path, go_annotations)
        serialized = json.dumps(report, ensure_ascii=False)
        print(serialized)
        if args.output_dir is not None:
            (args.output_dir / f"{gt_path.stem}_triplet_eval.json").write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
