"""Generate and validate incremental hierarchies using the training class splits.

Run from the repository root with ``python -m utils.create_hier --config ...``.
No API client is created unless a hierarchy actually needs generation.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import tempfile

from filelock import FileLock
import networkx as nx

BASE_PROMPT = """
You are a meticulous and highly accurate taxonomist and ontologist.
Your goal is to construct an initial hierarchical "is-a" taxonomy from a flat list of class names.

### INPUT
CLASS LIST (List):
A list of class names that must be organized into a hierarchical structure.

### NON-NEGOTIABLE RULES

1) TOTAL INCLUSION
- The final JSON must contain ALL class names from the CLASS LIST.
- Each input class must appear exactly once in the hierarchy.
- No omissions. No duplicates.

2) NAME PRESERVATION (CRITICAL)
- Keep every input class name EXACTLY as given (same spelling, spacing, and case).
- Do NOT rename input classes (e.g., do not convert spaces to underscores).
- Abstract parent nodes you create MAY use snake_case.

3) SINGLE ROOT
- All branches must ultimately connect to one root node:
  "entity"
- No disconnected subtrees are allowed.

4) TRUE "IS-A" RELATIONSHIPS
- Every parent → child relation must represent a genuine "is-a-type-of" relationship.
- Avoid vague placeholder parents such as:
  "object", "thing", "item", "concept"
  unless logically necessary.

5) MINIMAL BUT COHERENT ABSTRACTION
- Introduce abstract parent nodes only when necessary.
- Do not over-fragment the hierarchy.
- Prefer semantically meaningful intermediate categories.
- Do not introduce unnecessary multi-layer chains.

6) STRUCTURE CONSTRAINTS
- Output must be a single valid JSON object.
- Keys are parent nodes; values are lists of direct children.
- Only nodes with children appear as keys.
- Leaf nodes appear only in child lists.
- No empty lists.
- No cycles.
- No duplicate edges.
- No node may be a child of itself.

### OUTPUT FORMAT
- Return ONLY the valid JSON object.
- No markdown.
- No explanations.
- No comments.

### MANDATORY SELF-CHECK BEFORE OUTPUT
Before finishing, verify:

- Does every input class appear exactly once?
- Are all nodes connected (directly or indirectly) to "entity"?
- Are all relations valid "is-a" relations?
- Did I accidentally rename any input class?

If any answer is NO, fix the structure before output.

### Current Task

CLASS LIST:
{{CLASS_LIST}}
"""

INC_PROMPT = """
You are a meticulous and highly accurate taxonomist and ontologist.
Your goal is to update an existing hierarchical "is-a" structure by integrating a list of NEW class names into it.

### INPUTS
1) EXISTING HIERARCHY (JSON): current taxonomy adjacency list (parent -> [children])
2) NEW CLASSES (List): new class names that must be added

### NON-NEGOTIABLE RULES
1) TOTAL INCLUSION
- The final JSON must contain ALL strings already present anywhere in the EXISTING HIERARCHY (all keys and all children),
  AND ALL strings from NEW CLASSES.
- Do not delete anything.

2) NAME PRESERVATION (CRITICAL)
- For every class name that appears in NEW CLASSES, keep the name EXACTLY as given (same spacing/case).
- Do NOT rename input class names (e.g., do not change spaces to underscores).
- New abstract nodes you invent MAY use snake_case.

3) STABILITY / NO REWIRING
- You MUST NOT remove or change any existing parent->child relationship from the EXISTING HIERARCHY.
- You MUST NOT move an existing child from one parent list to another.
- You MAY ONLY:
  (a) append new children to existing parent lists, and/or
  (b) add new parent keys with their children lists, and/or
  (c) add new high-level children under "entity".
- Keep the single root node "entity".

4) MINIMAL NEW ABSTRACT NODES
- Create the minimum number of new abstract nodes required.

5) FALLBACK BUCKET
- If a NEW class cannot be confidently placed under an existing or newly created abstract parent, place it under:
  "unclassified" (create it if needed) which must be reachable from "entity".

### OUTPUT FORMAT
- Return ONLY one valid JSON object (no markdown, no comments).
- Keys are parents; values are lists of direct children.
- Only nodes with children are keys. No empty lists. No cycles. No duplicates.

### MANDATORY SELF-CHECK BEFORE OUTPUT
- Check that every NEW class appears exactly once in the final JSON (either as a child or as a key if it has children).
- Check that every string from the EXISTING HIERARCHY still appears in the final JSON.
- If any are missing, fix the JSON before finishing.

### Current Task
EXISTING HIERARCHY:
{{PREVIOUS_JSON_STRING}}

NEW CLASSES:
{{NEW_CLASS_LIST}}
"""


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_keys)


def _write_json(path, value):
    """Replace a file only after its complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_taxonomy(taxonomy, seen_classes, all_classes, previous=None):
    """Check class membership, tree structure, and preservation of the old tree."""
    if not isinstance(taxonomy, dict) or not taxonomy:
        return ["The hierarchy must be a non-empty JSON object."]
    errors = []
    graph = nx.DiGraph()
    for parent, children in taxonomy.items():
        if not isinstance(parent, str) or not parent:
            errors.append("Parent names must be non-empty strings.")
            continue
        graph.add_node(parent)
        if not isinstance(children, list) or not children:
            errors.append(f"{parent!r} must have a non-empty list of children.")
            continue
        if any(not isinstance(child, str) or not child for child in children):
            errors.append(f"{parent!r} contains invalid child names.")
            continue
        if len(children) != len(set(children)):
            errors.append(f"{parent!r} contains duplicate children.")
        graph.add_edges_from((parent, child) for child in children)
    if errors:
        return errors

    nodes = set(graph)
    expected = set(seen_classes)
    actual = nodes & set(all_classes)
    missing = expected - nodes
    future = actual - expected
    if missing:
        errors.append(f"Missing seen classes: {sorted(missing)}")
    if future:
        errors.append(f"Classes from future tasks must not appear: {sorted(future)}")
    if not nx.is_directed_acyclic_graph(graph):
        errors.append("The hierarchy contains a cycle or a self-edge.")
    roots = {node for node, degree in graph.in_degree() if degree == 0}
    if roots != {"entity"}:
        errors.append(f"The sole root must be 'entity', found {sorted(roots)}.")
    if "entity" not in nodes or nx.descendants(graph, "entity") | {"entity"} != nodes:
        errors.append("All nodes must be reachable from 'entity'.")
    multiple_parents = [node for node, degree in graph.in_degree() if degree > 1]
    if multiple_parents:
        errors.append(f"Nodes have multiple parents: {sorted(multiple_parents)}")
    if previous is not None:
        old_nodes = set(previous) | {
            child for children in previous.values() for child in children
        }
        old_edges = {
            (parent, child)
            for parent, children in previous.items()
            for child in children
        }
        if old_nodes - nodes:
            errors.append(f"Removed old nodes: {sorted(old_nodes - nodes)}")
        if old_edges - set(graph.edges):
            errors.append(
                f"Removed or rewired old edges: {sorted(old_edges - set(graph.edges))}"
            )
    return errors


def call_gpt(prompt, client, model="gpt-5.1"):
    with client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You must strictly follow the instructions and output ONLY valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        stream=True,
    ) as response:
        chunks = []
        for event in response:
            if event.choices and event.choices[0].delta.content:
                chunks.append(event.choices[0].delta.content)
    return json.loads("".join(chunks).strip(), object_pairs_hook=_unique_keys)


def generate_taxonomy(
    previous, new_classes, seen_classes, all_classes, client, model, max_fix_rounds=2
):
    if previous is None:
        prompt = BASE_PROMPT.replace(
            "{{CLASS_LIST}}", json.dumps(new_classes, ensure_ascii=False)
        )
    else:
        prompt = INC_PROMPT.replace(
            "{{PREVIOUS_JSON_STRING}}", json.dumps(previous, ensure_ascii=False)
        ).replace("{{NEW_CLASS_LIST}}", json.dumps(new_classes, ensure_ascii=False))
    base_prompt = prompt
    errors = []
    for attempt in range(max_fix_rounds + 1):
        try:
            taxonomy = call_gpt(prompt, client, model)
            errors = validate_taxonomy(taxonomy, seen_classes, all_classes, previous)
        except ValueError as error:
            taxonomy = None
            errors = [f"Invalid JSON response: {error}"]
        if not errors:
            return taxonomy
        logging.warning(
            "Hierarchy response rejected (attempt %s): %s",
            attempt + 1,
            "; ".join(errors),
        )
        prompt = (
            base_prompt
            + "\n### REPAIR THE PREVIOUS RESPONSE\n"
            + "Validation errors:\n"
            + "\n".join(errors)
            + "\nPrevious response:\n"
            + json.dumps(taxonomy, ensure_ascii=False)
            + "\nReturn a corrected JSON tree. Preserve every original node and edge, include all seen classes, and exclude future classes."
        )
    raise ValueError("Could not generate a valid hierarchy: " + "; ".join(errors))


def _task_classes(data_manager):
    batches = []
    start = 0
    for size in data_manager._increments:
        batch = data_manager._class_to_label[start : start + size]
        if size <= 0 or len(batch) != size:
            raise ValueError("Invalid task sizes or class labels in DataManager.")
        batches.append(list(batch))
        start += size
    flattened = [name for batch in batches for name in batch]
    if not batches or any(not isinstance(name, str) or not name for name in flattened):
        raise ValueError("Task classes must be non-empty lists of class names.")
    if len(flattened) != len(set(flattened)):
        raise ValueError("Class names must be unique across tasks.")
    return batches


def _valid_prefix(directory, batches, all_classes):
    """Return the valid stages before the first missing or inconsistent tree."""
    trees = []
    seen = []
    previous = None
    for task, batch in enumerate(batches):
        seen.extend(batch)
        path = directory / f"task_{task}.json"
        try:
            taxonomy = _read_json(path)
            errors = validate_taxonomy(taxonomy, seen, all_classes, previous)
        except (OSError, ValueError) as error:
            errors = [str(error)]
        if errors:
            return trees, f"{path}: " + "; ".join(errors)
        trees.append(taxonomy)
        previous = taxonomy
    return trees, None


def prepare_hierarchies(data_manager, args, *, check_only=False):
    """Reuse matching trees or generate a seed-specific cache before training."""
    batches = _task_classes(data_manager)
    all_classes = data_manager._class_to_label
    root = Path(args.get("knowledge_root", "knowledge"))
    step = data_manager._increments[1] if len(batches) > 1 else args["increment"]
    split = f"{len(batches[0])}_{step}"
    original_dir = root / data_manager.dataset_name / split
    seed = args["seed"]
    seed_name = (
        f"seed_{seed}" if isinstance(seed, int) else "seeds_" + "_".join(map(str, seed))
    )
    cache_dir = root / "generated" / data_manager.dataset_name / split / seed_name
    model = args.get("hierarchy_model") or os.getenv(
        "HASTEN_HIERARCHY_MODEL", "gpt-5.1"
    )

    # Checking alone has no write side effects and does not require the SDK/key.
    if check_only:
        for directory in (cache_dir, original_dir):
            trees, reason = _valid_prefix(directory, batches, all_classes)
            if reason is None:
                logging.info(
                    "All %s hierarchies match the current task classes: %s",
                    len(trees),
                    directory,
                )
                return directory
        raise ValueError(
            "No complete hierarchy matches the current task classes: " + reason
        )

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    # Each seed has its own cache; concurrent runs of that seed share a lock.
    with FileLock(str(cache_dir) + ".lock"):
        trees, cache_reason = _valid_prefix(cache_dir, batches, all_classes)
        if cache_reason is None:
            logging.info(
                "Reusing %s validated task hierarchies: %s", len(trees), cache_dir
            )
            return cache_dir
        original_trees, original_reason = _valid_prefix(
            original_dir, batches, all_classes
        )
        if original_reason is None:
            logging.info(
                "Reusing %s validated task hierarchies: %s",
                len(original_trees),
                original_dir,
            )
            return original_dir
        if len(original_trees) > len(trees) or (
            not cache_dir.exists() and original_dir.exists()
        ):
            trees = original_trees
            reason = original_reason
        else:
            reason = cache_reason
        logging.info("Hierarchy regeneration starts at task %s: %s", len(trees), reason)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Hierarchies do not match the current task classes. Set OPENAI_API_KEY "
                "to generate them, and OPENAI_BASE_URL if using an API proxy. " + reason
            )
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install the hierarchy-generation dependency: pip install openai"
            ) from error

        # API retry backoff must not perturb the training framework's random state.
        random_state = random.getstate()
        try:
            with OpenAI(timeout=180.0, max_retries=2) as client:
                cache_dir.mkdir(parents=True, exist_ok=True)
                seen = []
                previous = None
                for task, batch in enumerate(batches):
                    seen.extend(batch)
                    if task < len(trees):
                        taxonomy = trees[task]
                    else:
                        logging.info(
                            "Generating hierarchy for task %s (%s new classes)",
                            task,
                            len(batch),
                        )
                        taxonomy = generate_taxonomy(
                            previous,
                            batch,
                            seen,
                            all_classes,
                            client,
                            model,
                            max_fix_rounds=args.get("hierarchy_max_fix_rounds", 2),
                        )
                    _write_json(cache_dir / f"task_{task}.json", taxonomy)
                    previous = taxonomy
                _write_json(
                    cache_dir / "classes.json",
                    {
                        "dataset": data_manager.dataset_name,
                        "seed": seed,
                        "task_classes": batches,
                        "model": model,
                    },
                )
        finally:
            random.setstate(random_state)
        logging.info("Hierarchies ready: %s", cache_dir)
        return cache_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare task-wise trees using the training framework's class order."
    )
    parser.add_argument(
        "--config", required=True, help="The same experiment JSON used for training."
    )
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--model", help="Override hierarchy_model or HASTEN_HIERARCHY_MODEL."
    )
    parser.add_argument(
        "--knowledge-root", help="Override knowledge_root (default: knowledge)."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate existing trees without API calls or writes.",
    )
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = _read_json(options.config)
    settings["seed"] = options.seed
    if options.model:
        settings["hierarchy_model"] = options.model
    if options.knowledge_root:
        settings["knowledge_root"] = options.knowledge_root
    from utils.data_manager import DataManager

    data_manager = DataManager(
        settings["dataset"],
        settings["shuffle"],
        settings["seed"],
        settings["init_cls"],
        settings["increment"],
    )
    directory = prepare_hierarchies(
        data_manager, settings, check_only=options.check_only
    )
    print(f"Hierarchy directory: {directory}")


if __name__ == "__main__":
    main()
