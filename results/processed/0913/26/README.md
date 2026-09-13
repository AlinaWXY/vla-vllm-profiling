# 0913/26 — Failed before timing

PyTorch rejected replaying an existing vision graph while capturing a scope graph; the working directory also shadowed the requested snapshot.

No timing or roofline result was produced. Original log and source snapshot are preserved. The corrected baseline is 0913/27, which recaptures the same compiled vision callable and verifies its import path.
