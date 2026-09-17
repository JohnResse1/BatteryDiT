# Fixed SOC--OCV tables

This directory holds the fixed SOC--OCV tables used by `PhysicsOperator`.
Each local `.npz` contains only `soc` and `voltage`; it is created once from
the training split and reused by every training, validation and test run for
that dataset.

`extraction.py` contains the one-time shared extraction function. It is not a
training component and is not called by `main.py`.
