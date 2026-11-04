#!/bin/bash
# =====================================================================
#  SLURM driver used on the MSU Tempest cluster.  Usage (from this folder):
#     bash go.sh setup     # once: builds the Python environment (~5-10 min)
#     bash go.sh test      # ~10 min GPU self-test; wait for PASSED
#     bash go.sh run       # submits the full benchmark (36 jobs)
#     bash go.sh status    # how far along it is
#     bash go.sh collect   # makes stage2_results.zip to download
#  Stage 5:  bash go.sh test5  ->  bash go.sh run5  ->  bash go.sh status5  ->  bash go.sh grade5
#  Stage 6:  bash go.sh test6  ->  bash go.sh run6  ->  bash go.sh status6  ->  bash go.sh grade6
# =====================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV="$HOME/blcpinn_env"
RES="$HERE/results"
LOGS="$HERE/logs"
mkdir -p "$RES" "$LOGS"

pick_account() {
  # choose GPU partition + account automatically from your Slurm associations
  local accts
  accts=$(sacctmgr -nP show assoc user="$USER" format=account 2>/dev/null | sort -u)
  PRIO=$(echo "$accts" | grep '^priority-' | head -1)
  GRP=$(echo "$accts" | grep '^group-' | head -1)
  if [ -n "${FORCE_UNSAFE:-}" ] && [ -n "$GRP" ]; then PRIO=""; fi
  if [ -n "$PRIO" ]; then PART=gpupriority; ACCT=$PRIO
  elif [ -n "$GRP" ]; then PART=gpuunsafe; ACCT=$GRP
  else
    echo "Could not find a Tempest account for $USER. Accounts found: [$accts]"
    echo "Please check the Slurm account settings."; exit 1
  fi
  echo "Using partition $PART with account $ACCT"
}

load_python() {
  module purge >/dev/null 2>&1
  if python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then return 0; fi
  for m in Python Python/3.11 Python/3.10 Miniforge3 Mamba Anaconda3; do
    module load "$m" >/dev/null 2>&1 && \
      python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null && return 0
  done
  echo "No Python >= 3.9 found. Output of 'module avail Python':"; module avail Python 2>&1 | head -40
  echo "Please load a Python >= 3.9 module manually."; exit 1
}

case "${1:-}" in
setup)
  load_python
  echo "Python: $(python3 --version) at $(which python3)"
  if [ ! -d "$ENV" ]; then python3 -m venv "$ENV" || { echo "venv failed "; exit 1; }; fi
  source "$ENV/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install torch numpy || { echo "pip install failed "; exit 1; }
  python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
  echo; echo "SETUP DONE. Next: bash go.sh test"
  ;;
test)
  pick_account
  cat > "$HERE/test.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_test
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=$LOGS/test_%j.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
nvidia-smi --query-gpu=name,memory.total --format=csv
python bench2.py --selftest
python bench2.py --plan S2 --quick --task 0 --out "$HERE/quick_test" --only_methods pinn ae_plus gkpinn_pub gkpinn_hard fourier_s1 tsnn_g1
python bench2.py --plan S3 --quick --task 1 --out "$HERE/quick_test3"
python bench2.py --plan S3 --quick --task 18 --out "$HERE/quick_test3"
EOF
  jid=$(sbatch --parsable "$HERE/test.sbatch") || exit 1
  echo "Test job $jid submitted. Waiting for it to finish (usually 5-20 min; it may wait in the queue first)..."
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 20; printf "."; done; echo
  out="$LOGS/test_${jid}.out"
  tail -n 60 "$out"
  if grep -q "SELFTEST PASSED" "$out" && ! grep -q "FAILED" "$out"; then
    echo; echo "TEST PASSED. Next: bash go.sh run"
  else
    echo; echo "TEST HAD PROBLEMS. See the log file: $out"
  fi
  ;;
run)
  pick_account
  source "$ENV/bin/activate" 2>/dev/null || { echo "run 'bash go.sh setup' first"; exit 1; }
  N=$(python "$HERE/bench2.py" --plan S2 --n_groups 2>/dev/null | tail -1)
  cat > "$HERE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_s2
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-$((N-1))
#SBATCH --output=$LOGS/run_%A_%a.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
python bench2.py --plan S2 --task \$SLURM_ARRAY_TASK_ID --out "$RES"
EOF
  sbatch "$HERE/run.sbatch" && echo "Submitted $N jobs (each = one problem, one epsilon, one seed, all methods)." \
    && echo "You can log out now. Check progress any time with: bash go.sh status"
  ;;
status)
  done=$(ls "$RES"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  echo "Finished trainings: $done of 450"
  echo "Jobs still queued/running:"; squeue -u "$USER" -o "%.12i %.10P %.20j %.8T %.10M" 2>/dev/null
  grep -h "FAILED" "$LOGS"/run_*.out 2>/dev/null | head -5
  ;;
collect)
  cd "$HERE" && rm -f stage2_results.zip
  if command -v zip >/dev/null; then
    zip -qr stage2_results.zip results logs -x '*.tmp.npz' && f=stage2_results.zip
  else
    tar czf stage2_results.tar.gz --exclude='*.tmp.npz' results logs && f=stage2_results.tar.gz
  fi
  ls -lh "$f"
  echo "Download $f (folder: $HERE) from the Tempest Web 'Files' page."
  ;;
run3)
  # Stage 3: layer-balanced residual weighting test (convection problems only)
  pick_account
  source "$ENV/bin/activate" 2>/dev/null || { echo "run 'bash go.sh setup' first"; exit 1; }
  RES3="$HERE/results_s3"; mkdir -p "$RES3"
  N=$(python "$HERE/bench2.py" --plan S3 --n_groups 2>/dev/null | tail -1)
  cat > "$HERE/run3.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_s3
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-$((N-1))
#SBATCH --output=$LOGS/run3_%A_%a.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
python bench2.py --plan S3 --task \$SLURM_ARRAY_TASK_ID --out "$RES3"
EOF
  sbatch "$HERE/run3.sbatch" && echo "Submitted $N Stage-3 jobs. Check progress with: bash go.sh status3"
  ;;
status3)
  done=$(ls "$HERE/results_s3"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  total=$(source "$ENV/bin/activate" 2>/dev/null; python "$HERE/bench2.py" --plan S3 --n_runs 2>/dev/null | tail -1)
  echo "Stage 3 finished trainings: $done of $total"
  echo "Jobs still queued/running:"; squeue -u "$USER" -o "%.12i %.10P %.20j %.8T %.10M" 2>/dev/null
  grep -h "FAILED" "$LOGS"/run3_*.out 2>/dev/null | head -5
  ;;
grade3)
  source "$ENV/bin/activate" && cd "$HERE" && python grade.py results_s3 refs_on_eval_grid.npz stage3 \
    && echo "Download stage3_table.txt and stage3_metrics.json."
  ;;
run4)
  # Stage 4: (a) final method on Ex 4 (3 jobs), (b) one sequential timing job on a single GPU
  pick_account
  source "$ENV/bin/activate" 2>/dev/null || { echo "run 'bash go.sh setup' first"; exit 1; }
  for P in S4 S4T; do
    N=$(python "$HERE/bench2.py" --plan $P --n_groups 2>/dev/null | tail -1)
    cat > "$HERE/run_$P.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_$P
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-$((N-1))
#SBATCH --output=$LOGS/run_${P}_%A_%a.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
python bench2.py --plan $P --task \$SLURM_ARRAY_TASK_ID --out "$HERE/results_$P"
EOF
    sbatch "$HERE/run_$P.sbatch" && echo "Submitted plan $P ($N job(s))."
  done
  echo "Check progress with: bash go.sh status4"
  ;;
status4)
  a=$(ls "$HERE/results_S4"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  b=$(ls "$HERE/results_S4T"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  echo "Stage 4 finished: part A $a of 9, part B (timing) $b of 28"
  echo "Jobs still queued/running:"; squeue -u "$USER" -o "%.12i %.10P %.20j %.8T %.10M" 2>/dev/null
  grep -h "FAILED" "$LOGS"/run_S4*.out 2>/dev/null | head -5
  ;;
grade4)
  source "$ENV/bin/activate" && cd "$HERE" \
    && python grade.py results_S4 refs_on_eval_grid.npz stage4 \
    && python grade.py results_S4T refs_on_eval_grid.npz stage4t allmaps \
    && echo "Download stage4_metrics.json, stage4t_metrics.json and stage4t_figdata.npz."
  ;;
test5)
  # Stage 5 quick check (~10-20 min): self-test + tiny versions of both new experiments
  pick_account
  cat > "$HERE/test5.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_t5
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=$LOGS/test5_%j.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
nvidia-smi --query-gpu=name,memory.total --format=csv
python bench2.py --selftest
python bench2.py --plan S5 --quick --task 0 --out "$HERE/quick_test5"
python bench2.py --plan S5 --quick --task 12 --out "$HERE/quick_test5"
python bench2.py --plan S5T --quick --task 0 --out "$HERE/quick_test5t" --only_problems conv_var
EOF
  jid=$(sbatch --parsable "$HERE/test5.sbatch") || exit 1
  echo "Test job $jid submitted. Waiting for it to finish (usually 10-20 min; it may wait in the queue first)..."
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 20; printf "."; done; echo
  out="$LOGS/test5_${jid}.out"
  tail -n 40 "$out"
  nq=$(ls "$HERE"/quick_test5/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  if grep -q "SELFTEST PASSED" "$out" && ! grep -q "FAILED" "$out" && [ "$nq" -ge 7 ]; then
    echo; echo "TEST PASSED. Next: bash go.sh run5"
  else
    echo; echo "TEST HAD PROBLEMS. See the log file: $out"
  fi
  ;;
run5)
  # Stage 5: (a) product-form BC inside BLC-PINN, (b) competitors with 4x budget, (c) one timing job
  pick_account
  source "$ENV/bin/activate" 2>/dev/null || { echo "run 'bash go.sh setup' first"; exit 1; }
  for P in S5 S5T; do
    N=$(python "$HERE/bench2.py" --plan $P --n_groups 2>/dev/null | tail -1)
    cat > "$HERE/run_$P.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_$P
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-$((N-1))
#SBATCH --output=$LOGS/run_${P}_%A_%a.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
python bench2.py --plan $P --task \$SLURM_ARRAY_TASK_ID --out "$HERE/results_$P"
EOF
    sbatch "$HERE/run_$P.sbatch" && echo "Submitted plan $P ($N job(s))."
  done
  echo "Check progress with: bash go.sh status5"
  ;;
status5)
  a=$(ls "$HERE/results_S5"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  b=$(ls "$HERE/results_S5T"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  echo "Stage 5 finished: main part $a of 84, timing part $b of 8"
  echo "Jobs still queued/running:"; squeue -u "$USER" -o "%.12i %.10P %.20j %.8T %.10M" 2>/dev/null
  grep -h "FAILED" "$LOGS"/run_S5*.out 2>/dev/null | head -5
  ;;
grade5)
  source "$ENV/bin/activate" && cd "$HERE" \
    && python grade.py results_S5 refs_on_eval_grid.npz stage5 \
    && python grade.py results_S5T refs_on_eval_grid.npz stage5t \
    && echo "Download stage5_metrics.json and stage5t_metrics.json."
  ;;
regrade)
  # re-grade all saved runs against the improved Ex. 2 reference (refs_on_eval_grid_v2.npz); no retraining
  [ -f "$HERE/refs_on_eval_grid_v2.npz" ] || { echo "Upload refs_on_eval_grid_v2.npz into $HERE first"; exit 1; }
  source "$ENV/bin/activate" && cd "$HERE" \
    && python grade.py results    refs_on_eval_grid_v2.npz stage2r \
    && python grade.py results_s3 refs_on_eval_grid_v2.npz stage3r \
    && python grade.py results_S4 refs_on_eval_grid_v2.npz stage4r \
    && python grade.py results_S4T refs_on_eval_grid_v2.npz stage4tr allmaps \
    && python grade.py results_S5 refs_on_eval_grid_v2.npz stage5r \
    && python grade.py results_S5T refs_on_eval_grid_v2.npz stage5tr \
    && rm -f regraded.zip \
    && (command -v zip >/dev/null && zip -q regraded.zip stage2r_metrics.json stage3r_metrics.json stage4r_metrics.json stage4tr_metrics.json stage4tr_figdata.npz stage5r_metrics.json stage5tr_metrics.json \
        || tar czf regraded.tar.gz stage2r_metrics.json stage3r_metrics.json stage4r_metrics.json stage4tr_metrics.json stage4tr_figdata.npz stage5r_metrics.json stage5tr_metrics.json) \
    && ls -lh regraded.* && echo "Download regraded.zip (or regraded.tar.gz)."
  ;;
test6)
  # Stage 6 quick check (~10-20 min): self-test + tiny versions of the new runs
  pick_account
  cat > "$HERE/test6.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_t6
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=$LOGS/test6_%j.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
nvidia-smi --query-gpu=name,memory.total --format=csv
python bench2.py --selftest
python bench2.py --plan S6 --quick --task 0 --out "$HERE/quick_test6"
EOF
  jid=$(sbatch --parsable "$HERE/test6.sbatch") || exit 1
  echo "Test job $jid submitted. Waiting for it to finish (usually 10-20 min; it may wait in the queue first)..."
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 20; printf "."; done; echo
  out="$LOGS/test6_${jid}.out"
  tail -n 40 "$out"
  nq=$(ls "$HERE"/quick_test6/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  if grep -q "SELFTEST PASSED" "$out" && ! grep -q "FAILED" "$out" && [ "$nq" -ge 8 ]; then
    echo; echo "TEST PASSED. Next: bash go.sh run6"
  else
    echo; echo "TEST HAD PROBLEMS. See the log file: $out"
  fi
  ;;
run6)
  # Stage 6: two-scale NN with the published penalty boundary treatment (+ one timing job for all methods)
  pick_account
  source "$ENV/bin/activate" 2>/dev/null || { echo "run 'bash go.sh setup' first"; exit 1; }
  for P in S6 S6T; do
    N=$(python "$HERE/bench2.py" --plan $P --n_groups 2>/dev/null | tail -1)
    cat > "$HERE/run_$P.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=blcpinn_$P
#SBATCH --account=$ACCT
#SBATCH --partition=$PART
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-$((N-1))
#SBATCH --output=$LOGS/run_${P}_%A_%a.out
$(declare -f load_python)
load_python
source "$ENV/bin/activate"
cd "$HERE"
python bench2.py --plan $P --task \$SLURM_ARRAY_TASK_ID --out "$HERE/results_$P"
EOF
    sbatch "$HERE/run_$P.sbatch" && echo "Submitted plan $P ($N job(s))."
  done
  echo "Check progress with: bash go.sh status6"
  ;;
status6)
  a=$(ls "$HERE/results_S6"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  b=$(ls "$HERE/results_S6T"/*.npz 2>/dev/null | grep -v '\.tmp\.' | wc -l)
  echo "Stage 6 finished: main part $a of 96, timing part $b of 24"
  echo "Jobs still queued/running:"; squeue -u "$USER" -o "%.12i %.10P %.20j %.8T %.10M" 2>/dev/null
  grep -h "FAILED" "$LOGS"/run_S6*.out 2>/dev/null | head -5
  ;;
grade6)
  [ -f "$HERE/refs_on_eval_grid_v2.npz" ] || { echo "refs_on_eval_grid_v2.npz is missing in $HERE"; exit 1; }
  source "$ENV/bin/activate" && cd "$HERE" \
    && python grade.py results_S6 refs_on_eval_grid_v2.npz stage6 \
    && python grade.py results_S6T refs_on_eval_grid_v2.npz stage6t allmaps \
    && rm -f stage6.zip stage6.tar.gz \
    && (command -v zip >/dev/null && zip -q stage6.zip stage6_metrics.json stage6t_metrics.json stage6t_figdata.npz \
        || tar czf stage6.tar.gz stage6_metrics.json stage6t_metrics.json stage6t_figdata.npz) \
    && ls -lh stage6.* && echo "Download stage6.zip (or stage6.tar.gz)."
  ;;
*)
  sed -n '2,10p' "$0"
  ;;
esac
