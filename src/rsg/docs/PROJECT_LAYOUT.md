# Project layout

`nodes/` contains the three node entrypoints. Python helper modules are grouped
under `nodes/support/` by the node that owns them. The C++ fuser is a single
source file so future risk annotation inputs can be added without modifying
Phase 1.

All runtime parameters remain in `config/`, launch descriptions in `launch/`,
and the runtime configuration and launch files.
