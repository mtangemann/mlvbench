"""Launch the full experiment locally or on a SLURM cluster."""

import argparse
import logging
import shutil
import subprocess
import typing
from datetime import date, datetime
from pathlib import Path

import coloredlogs
import submitit
import yaml
from experiment import Condition, Config, run, warmup
from tabulate import tabulate

from mlvbench.models import list_models

LOGGER = logging.getLogger("launch.py")

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser(prog="launch.py")
parser.add_argument(
    "command",
    choices=("launch", "status", "cancel"),
    nargs="?",
    default="launch",
)
parser.add_argument(
    "--workspace",
    type=Path,
    default=ROOT / "output" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
parser.add_argument(
    "--config",
    type=Path,
    default=ROOT / "configs" / "default.yaml",
)
parser.add_argument("--retry", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--warmup", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)

ARGS = parser.parse_args()

# If SLURM is not available, we run all jobs sequentially in-process. This affects the
# behavior of this script in several places, so we set a global SLURM switch.
SLURM = not ARGS.debug and submitit.AutoExecutor.which() == "slurm"
if not SLURM:
    print("SLURM not found; running jobs locally.")

EXECUTOR = submitit.AutoExecutor(
    folder=ARGS.workspace / "slurm",
    cluster="debug" if ARGS.debug else None,
)


# -------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------

def get_output_path(workspace: Path, config: Config) -> Path:
    """Return the output path for a configuration within a workspace.

    Args:
        workspace: The path of the workspace.
        config: The configuration to get the output path for.

    Returns:
        The path of the directory that holds all outputs of the corresponding job.
    """
    weights = "pretrained" if config.pretrained else f"random_{config.model_seed}"
    model = config.model.replace("/", "_")
    return workspace / "results" / model / weights / config.condition


def resolve_models(specs: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for spec in specs:
        models = list_models(bundle=spec) if "/" not in spec else [spec]
        for model in models:
            if model not in seen:
                seen.add(model)
                result.append(model)
    return result


def load_configs() -> list[Config]:
    with open(ARGS.config) as f:
        launch_config = yaml.safe_load(f)

    conditions = launch_config.get("conditions", typing.get_args(Condition))
    models = resolve_models(launch_config["models"])
    base_overrides = launch_config.get("base", {})
    model_overrides = launch_config.get("model_overrides", {})

    configs = []

    for model in models:
        for condition in conditions:
            overrides = {**base_overrides, **model_overrides.get(model, {})}
            configs.append(Config(model=model, condition=condition, **overrides))

    return configs


CONFIGS = load_configs()


# -------------------------------------------------------------------------------------
# Status
# -------------------------------------------------------------------------------------

def get_active_jobs() -> dict[str, dict[str, str]]:
    """Return all jobs that are currently running or queued.

    - On SLURM, jobs submitted by this script include a comment with their output path,
      so that we can identify jobs belonging to this experiment.
    - When running locally, we assume that no job is running and emit a warning.
    """
    jobs = {}

    if not SLURM:
        print("Warning: Cannot find active jobs locally.")
        return jobs

    squeue_command = ["squeue", "--me", r"--format=%k %T %i %R"]
    squeue_output = subprocess.check_output(squeue_command).decode().split("\n")

    for line in squeue_output:
        if line.startswith(str(ARGS.workspace)):
            comment, state, slurm_id, *reason_parts = line.split()
            jobs[comment] = {
                "state": state.lower(),
                "slurm_id": slurm_id,
                "reason": " ".join(reason_parts),
            }

    return jobs


def get_status() -> dict[str, dict[str, str]]:
    status = {}

    active_jobs = get_active_jobs()

    warmup_job = active_jobs.get(str(ARGS.workspace / "<warmup>"), None)
    if warmup_job is not None:
        status["<warmup>"] = {
            "model": "<warmup>",
            "weights": "-",
            "condition": "-",
            "slurm_id": warmup_job["slurm_id"],
            "state": warmup_job["state"],
            "reason": warmup_job["reason"],
        }

    for config in CONFIGS:
        output_path = get_output_path(ARGS.workspace, config)
        active_job = active_jobs.get(str(output_path))

        if (output_path / "done").exists():
            with open(output_path / "done", "r") as f:
                state = f.read().strip()
        elif active_job is not None:
            state = active_job["state"]
        else:
            state = "ready"

        relative_path = output_path.relative_to(ARGS.workspace)
        status[str(relative_path)] = {
            "model": config.model,
            "weights": relative_path.parent.name,
            "condition": config.condition,
            "slurm_id": active_job["slurm_id"] if active_job else None,
            "state": state,
            "reason": active_job["reason"] if active_job else "",
        }

    return status


if ARGS.command == "status":
    print(tabulate(get_status().values(), headers="keys", tablefmt="simple"))
    exit()


# -------------------------------------------------------------------------------------
# Cancel
# -------------------------------------------------------------------------------------

if ARGS.command == "cancel":
    active_jobs = get_active_jobs()
    for job in active_jobs.values():
        subprocess.run(["scancel", job["slurm_id"]])
    print(f"Cancelled {len(active_jobs)} jobs")
    exit()


# -------------------------------------------------------------------------------------
# Workspace
# -------------------------------------------------------------------------------------

def write_workspace_readme() -> None:
    """Create a README stub for the workspace, unless it exists already.

    The README serves as experiment description and logbook and is maintained manually
    afterwards. Existing READMEs are therefore never modified.
    """
    readme_path = ARGS.workspace / "README.md"
    if readme_path.exists():
        return

    lines = [
        f"# {ARGS.workspace.name}",
        "",
        "TODO Describe the purpose of this run.",
        "",
        "## Logbook",
        "",
    ]

    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))


write_workspace_readme()


# -------------------------------------------------------------------------------------
# Warmup
# -------------------------------------------------------------------------------------

submitted_jobs = 0

EXECUTOR.update_parameters(
    slurm_partition="gpubase_l40s_b1",
    cpus_per_task=8,
    mem_gb=32,
    gpus_per_node=0,
    timeout_min=180,
    stderr_to_stdout=True,
)

def warmup_all() -> None:
    for config in CONFIGS:
        warmup(config)

# Skip the warmup for local runs. Since jobs are run sequentially, the warmup is pure
# overhead in this case.
warmup_job = None
if ARGS.warmup and SLURM:
    EXECUTOR.update_parameters(
        slurm_additional_parameters={"comment": str(ARGS.workspace / "<warmup>")}
    )
    warmup_job = EXECUTOR.submit(warmup_all)
    if ARGS.debug:
        warmup_job.wait()
    submitted_jobs += 1


# -------------------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------------------

EXECUTOR.update_parameters(
    slurm_partition="gpubase_l40s_b1",
    cpus_per_task=8,
    mem_gb=32,
    gpus_per_node=1,
    timeout_min=180,
    stderr_to_stdout=True,
)

def run_single(config: Config, output_path: Path) -> None:
    # The job owns all outputs below its directory (the trained probe and all
    # evaluations), so any previous outputs are removed before rerunning it.
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    try:
        run(config, output_path)

    except Exception as error:
        with open(output_path / "done", "w") as f:
            f.write("failed\n")
        raise error

    else:
        with open(output_path / "done", "w") as f:
            f.write("completed\n")


# Local sequential execution: run every pending config one-by-one in this process.
# Unlike the standard submit local executor this stays in the foreground (so Ctrl+C
# works) and, unlike --debug, it does not drop into a post-mortem debugger. A failing
# run is marked "failed" (by run_single) and execution continues.
if not SLURM:
    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")

    status = get_status()
    submit_states = ["ready", "failed"] if ARGS.retry else ["ready"]

    for config in CONFIGS:
        output_path = get_output_path(ARGS.workspace, config)
        relative_path = str(output_path.relative_to(ARGS.workspace))

        if status[relative_path]["state"] not in submit_states:
            continue

        LOGGER.info("Running %s ...",  relative_path)
        try:
            run_single(config, output_path)
        except Exception:
            LOGGER.exception("Run failed: %s", relative_path)

    exit()

if warmup_job is not None and not ARGS.debug:
    warmup_dependency = {"dependency": f"afterok:{warmup_job.job_id}"}
else:
    warmup_dependency = {}

status = get_status()
submit_states = ["ready", "failed"] if ARGS.retry else ["ready"]

for config in CONFIGS:
    output_path = get_output_path(ARGS.workspace, config)
    relative_path = str(output_path.relative_to(ARGS.workspace))

    if status[relative_path]["state"] not in submit_states:
        continue

    EXECUTOR.update_parameters(
        slurm_additional_parameters={**warmup_dependency, "comment": str(output_path)}
    )

    job = EXECUTOR.submit(run_single, config, output_path)
    if ARGS.debug:
        job.wait()

    submitted_jobs += 1

print(f"Submitted {submitted_jobs} jobs")
