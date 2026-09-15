# Patches to vendored third-party code

`third_party/` is gitignored, so every modification we make to UniVLA's checkout lives here as a diff and must be
re-applied after a fresh clone.  `REPRODUCE.md` lists the clone step; apply these immediately after it:

    cd third_party/UniVLA && git apply ../../acdc_la/patches/*.patch

Each patch states what it changes and why in its own header comment inside the diff.
