"""ACDC-LA: Preserve (AC/DC temporal factorisation) -> Disentangle (TC/TI) -> Transport (LATP).

See README.md for the paper line, the per-stage metrics and how to run.  Module map:
    paths                  filesystem constants
    data.oxe_mix           multi-source OXE clip dataset with UniVLA's sampling (train/val, action pairs)
    data.distractor        moving-patch nuisance rendering (the exogenous-motion intervention)
    common.features        frozen DINOv2 g4 features, quadrants, DCT temporal basis, effect cells
    common.model           Gaussian tokenizer E/U, state-conditioned transformer decoder, KL/MMD helpers
    common.probes          per-source episode-disjoint ridge probes (the only supervised code in the repo)
    stage1_preserve        tokenizer training (rec + KL) and the preservation probes
    stage2_disentangle     TC/TI split over the AC/DC cells with the nuisance intervention; G_sep
    stage3_transport       LATP: frozen tokenizer, transport operator trained by swap realism; readers
    baselines.univla       UniVLA stage-2 latent dump on the same val clips (for every stage's comparison)
"""
