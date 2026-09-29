# For AI agents working in this repo

- Each model is one folder: `dgx-spark/<model>/` or `mac/<model>/`, with the same scripts
  (`setup.sh`, `pull.sh`, `start.sh`, `chat.sh`, `bench.sh`, `status.sh`, `logs.sh`, `stop.sh`) and a
  `recipe.env` holding every setting. Change settings in `recipe.env` or per run (`NODES=1 ./start.sh`), not in `lib/`.
- The scripts all call `lib/recipe.sh`, which loads `lib/common.sh`, `config/cluster.env` (Sparks) and one
  engine file: `lib/engine-tensorfold-spark.sh` or `lib/engine-tensorfold-mac.sh`.
- DGX Spark scripts run on Spark 1 and reach Spark 2 over SSH. Run `./tools/doctor.sh` first.
- One model at a time per Spark. Stop the running recipe before starting another.
- Measure with `./bench.sh` and compare with `python3 bench/compare.py`; never claim a speed you did not measure.
- Versions are pinned (`lib/common.sh`). The Spark image builds TensorFold from `engine/` (our `glm-long-context`
  branch on 0.3.6.3; `TENSORFOLD_SOURCE=pinned` for upstream 0.3.6.3). Engine changes need their CUDA tests on a Spark
  (`tests/cuda/test_glm_*.py`, `test_flashnext_*.py`) and must keep drafted replies byte-identical to serial ones.
- `lib/common.sh` and `lib/engine-tensorfold-mac.sh` must stay bash 3.2 compatible (macOS).
- TensorFold only: no vLLM, Ollama or other engines in this project (other engines may only be benchmarked
  from the outside as a baseline, never run by it).
- GLM's DFlash2 draft model is non-commercial (CC BY-NC-ND 4.0): only use it after the user opts in.
