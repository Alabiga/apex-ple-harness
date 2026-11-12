# Apex-PLE: Constraint-Gated Agent Harness for Physical-Engineering's Last Exam

[![Benchmark](https://img.shields.io/badge/Benchmark-PLE--85_Tasks-blue.svg)](https://ple-bench.com/)
[![Track](https://img.shields.io/badge/Kaggle-Physical--Engineering%27s_Last_Exam-gold.svg)](https://www.kaggle.com/competitions/physical-engineerings-last-exam)
[![Prize](https://img.shields.io/badge/Prize_Pool-$400,000_USD-green.svg)](https://www.kaggle.com/competitions/physical-engineerings-last-exam)
[![Protocol](https://img.shields.io/badge/Protocol-PLE_Wire_v1_(WebSocket+MCP)-orange.svg)](#wire-protocol-specification)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Apex-PLE** is a production-grade, constraint-gated autonomous engineering harness designed specifically for the **Google - Agent Harness on Physical-Engineering's Last Exam (PLE)** benchmark.

---

## 1. Why Naive LLM Agents Fail on PLE

Constrained engineering tasks in PLE are evaluated under the strict mathematical scoring law:

$$s = 0.4 \cdot p + 0.6 \cdot G \cdot O'$$

where:
- $p \in [0, 1]$ is the fraction of satisfied physical, structural, or code constraints.
- $G \in \{0, 1\}$ is the **Binary Feasibility Gate**:
  $$G = 1 \iff p = 1.0$$
  If even a single hard safety or design code constraint is violated by $10^{-4}$, $G = 0$, immediately forfeiting **60% of the entire task score**.
- $O' = \text{clip}\left(\frac{O - O_{\text{naive}}}{1 - O_{\text{naive}}}, 0, 1\right)$ is the normalized objective quality relative to certified bounds.

Naive agents fail because they optimize the objective $O$ greedily before securing feasibility, pushing decisions right against constraint boundaries where floating-point or model errors drop $G \to 0$.

Apex-PLE solves this with a **feasibility-first, interior-point optimization paradigm**:
1. It builds a dedicated Python solver script inside the container with a **2%–5% conservative safety buffer** ($g_i(x) \le -\epsilon_{\text{safe}}$).
2. The instant $p = 1.0$ is certified, it snapshots deliverables to `/tmp/.apex_ple_checkpoint`.
3. Objective optimization proceeds strictly inside the feasible region, only overwriting deliverables if the auditor verifies $G = 1$.

---

## 2. Five-Phase Engineering Architecture

```
[ Inbound "task" Frame ]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: Reconnaissance & Dynamic MCP Discovery        │
│ • Container directory scan (/app)                      │
│ • Streamable-HTTP JSON-RPC MCP initialization          │
│ • Discipline classifier (Structural/Civil/Elec/Mech)   │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Feasibility-First Synthesis (G = 1 Gate)      │
│ • Conservative calculation with safety buffer          │
│ • Automatic deliverable generation                     │
│ • Atomic snapshot to /tmp/.apex_ple_checkpoint         │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Interior-Point Constrained Optimization (O')  │
│ • Bounded numerical optimization (scipy/MILP/grid)     │
│ • Overwrite ONLY when G = 1 holds                      │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Deterministic Deliverable Auditor             │
│ • File existence, non-zero byte size                   │
│ • Deep integrity check: JSON/YAML/CSV/XLSX/DXF/IFC/MD  │
│ • Strict NaN / Inf / formula-cache error interception  │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Rubric-Aligned Engineering Executive Report   │
│ • 5-section report: standards, manifest, matrix, O'    │
│ • Outbound terminal {"type": "done"} frame             │
└────────────────────────────────────────────────────────┘
```

---

## 3. Wire Protocol Specification (`serve.py`)

The harness operates as a WebSocket server on `0.0.0.0:$PORT` (default `8080`), accepting connection from the official PLE evaluator container.

### Inbound Frames:
- `task`: Task specification with `session`, `instruction`, `workdir`, `limits`, and `mcp_servers`.

### Outbound Action Frames & Inbound Results:
- `exec`: Execute container shell commands (`stdout`, `stderr`, `exit_code`). Head+tail smart truncation protects context limits.
- `read_file`: Read binary/text deliverables.
- `write_file`: Atomically write deliverables. Payloads larger than `max_write_bytes` are safely streamed.
- `http`: Proxy MCP JSON-RPC 2.0 calls directly to local container MCP servers.
- `done`: Emits final rubric report and concludes task evaluation.

---

## 4. Submission Configuration (`submission.yaml`)

Conforms strictly to the official PLE schema:

```yaml
name: apex-ple-harness
model_api: google-cloud/claude-opus-5-5
install: >-
  uv venv --python 3.12 /opt/harness/venv
  && uv pip install --python /opt/harness/venv/bin/python -r {src}/requirements.txt
serve: /opt/harness/venv/bin/python {src}/serve.py
env:
  LOG_LEVEL: "info"
  REASONING_EFFORT: "high"
```

Variants are also provided:
- `submission_argon.yaml` (`model_api: google-cloud/gemini-4-argon`)
- `submission_flash.yaml` (`model_api: google-cloud/gemini-3-8-flash`)

---

## 5. Local Testing & Verification

Apex-PLE includes a deterministic offline solver mode (`APEX_OFFLINE_SOLVER=1`) allowing complete end-to-end testing without cloud API keys or external credits:

```bash
# Run static validation check
python scripts/validate_ple_submission.py

# Run full E2E benchmark simulator across all 4 disciplines
python scripts/test_ple_e2e.py
```
