# basalt-reference — external donor reference clone

Read-only reference repo for the port's cross-engine reading; intentionally NOT a
submodule or gitlink (ignored in the parent tree since 2026-11-27).

- Remote: https://github.com/jesdga95/basalt
- Pinned commit: 34f4c3bb1079f1b006620684cbddd5c8b5d6280f
- Restore: `git clone https://github.com/jesdga95/basalt port/basalt-reference && git -C port/basalt-reference checkout 34f4c3bb`
- Used by: `port/m3/graft-checklist.md` (engine-surface cross-check) and graft type reading.
