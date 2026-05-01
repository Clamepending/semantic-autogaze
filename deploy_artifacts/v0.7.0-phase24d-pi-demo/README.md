# v0.7.0-phase24d-pi-demo

Pi 5 deploy bundle for the phase24d ConvNeXt-atto scorer (current openvocab
leader at composite +22 vs v0.6.0's +17, same Pi latency).

## Files

| file                                                         | sha256 | size |
|--------------------------------------------------------------|--------|------|
| `phase24d_convnext_atto_mpp05_aggrAug_best_v070.pt`          | _(compute on upload)_ | 13.4 MB |

## Recipe (training-side, frozen at deploy)

- Backbone: `convnext-atto.d2_in1k` (timm, 3.7M params, ImageNet-pretrained)
- Head: `TextScorerHead` (14×14 grid, per-query bias MLP)
- SiglipBias: `SiglipBiasPerQuery` (Linear 512→64, GELU, Linear 64→1, init bias=-2.0)
- Trained 10K fine-tune from `results/phase19b_atto_perquery_10k/best_val.pt` (v0.6.0)
- Training flags: `--per_query_bias --augment --augment_aggressive --multi_prompt_training --multi_prompt_p 0.5 --bce_pos_weight 30 --fn_filter --balanced_pos_frac 0.6 --source_weights "pp:5,stuff:1.5,lvis:1,coco:1,ade20k:1.5" --distill_teacher_ckpt phase13_dinov2s_long_bestval --lambda_distill 0.5 --lr 5e-4 --batch_size 16`

Training-time-only flags (`--multi_prompt_training`, `--augment_aggressive`,
`--distill_teacher_ckpt`) have **zero deployment cost**: at inference the
head sees a frozen ConvNeXt-atto backbone + per-query MLP bias, identical
to v0.6.0.

## Pi 5 deploy

```bash
# fresh install path
curl -L https://raw.githubusercontent.com/Clamepending/semantic-autogaze/main/demo/pi_setup.sh -o pi_setup.sh
bash pi_setup.sh                              # MODEL defaults to phase24d-atto

# explicit model selection
MODEL=phase24d-atto bash pi_setup.sh

# upgrade in place from v0.6.0 install
cd ~/semantic-autogaze-pi-demo
rm -f phase19b_convnext_atto_perquery_best_v060.pt
MODEL=phase24d-atto bash pi_setup.sh          # re-runs with new ckpt
```

Browser at `http://<pi-host>:8000/` shows the live MJPEG stream + keyword
input + threshold slider. Default threshold 0.45 absolute sigmoid.

## Verification on the Pi

After first launch the server logs:

```
[model] phase24d-atto from phase24d_convnext_atto_mpp05_aggrAug_best_v070.pt
[ckpt args] model=convnext-atto per_query_bias=True per_query_bias_kind=mlp ...
[sb] using SiglipBiasPerQuery (MLP 512->64->1, init bias=-2.0)
```

If you see `using SiglipBiasPerQuery` the per-query MLP loaded correctly.
A fallback to `using SiglipBias (global)` indicates the ckpt was loaded
into the v0.5.0-style head — the bias_mlp keys would be missing. The
`pi_webcam_server.py` shipped here detects `bias_mlp.*` automatically.

## Failure modes still in this ckpt (from `OPENVOCAB_REFLECTION.md`)

- **Indoor close-up + Pi-on-chest down-view scenes still hit the
  horizon-band spatial prior.** ConvNeXt-atto's ImageNet inductive bias +
  unified-targets-AB supervision distribution. Fix candidates are
  backbone-level only: `phase25_dinov2s_recipe_best` breaks the prior
  (composite +18 but better localization on indoor) at 4-5x Pi 5 latency.
- Compositional adjective queries (`red sign`, `white chair`) under-fire
  even when spatially correct (CLIP text-encoder bottleneck; not addressable
  at deploy without TTA, which was tested and didn't help on common queries).

## Provenance

- Code branch: `r/phase10-atto-egoschema-vqa @ <commit>` for the demo bundle.
- Training branch: `r/phase24d_atto_mpp05_aggrAug_10k` produced the ckpt
  via the `launch_phase24_sweep.sh` design-space sweep.
- Cross-model leaderboard reproduction:
  `bash scripts/eval_openvocab_parallel.sh && python -m scripts.compare_openvocab_sweep`.
