# M2 reference server and evidence (closed 2026-11-24)

FP32 torch mirror of the Inkling-2-Mini decode math — the oracle used to root-cause the
M2 anchored-parity failure. Conclusion (full record: `../m0/parity/README.md` parity
contract + `Projects/Inkling-2-Mini.md` in the vault): the miss is greedy-trajectory
chaos (FP32 matmul tiling ~1e-7, amplified ~2.5x per visit x 48 visits), not a math
bug; the mirror reproduces the donor int8 server bit-for-bit once it mirrors the
engine's Int8G64 full-layer KV coding (`INK2_KV_CODING=int8`, the default here).

Canonical files:
- `reference.py` — the mirror server (port 8193). The int8-coding fix (originally
  commit 8567b8ce) is in this working copy; the commit itself was dropped by the
  2026-11-24 branch reset to dd80950f and re-landed in the port checkpoint commit.
- `m2_results.json`, `m2_anchored*.json` — probe outcomes (F16 and int8 mirrors both
  miss at step 0 on long anchors; length bisect matches the int8 server at 28-32 tokens).
- `probe_compare.py`, `probe_anchored.py`, `probe_step0.py`, `probe_bisect.py` —
  comparison harnesses against the donor oracle (port 8191) and this mirror.
- `traces/` — per-layer trace dumps from the root-cause session (port vs donor F16 and
  int8-KV servers), moved from `.agent-tmp/` (ignored scratch) for durability 2026-11-27.
- `m2_step0.log`, `m2_step0_int8.log`, `m2_bisect.log`, `inv_dbg.txt`, `expert7.pt` —
  raw session logs kept as experiment evidence; do not prune.
