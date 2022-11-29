# Mechanistic Interpretability: Code Generation

## Getting the data

Note this assumes your tmp directory has space for c.150GB of data, and that
your working directory is `/workspaces/mech-interp-code-generation` (i.e. change
appropriately for your use case).

```bash
mkdir /tmp/data
git clone https://huggingface.co/NeelNanda/full_pred_log_probs -c /tmp/data/mech-interp-code-generation
ln -s /tmp/data/ /workspaces/mech-interp-code-generation/data
```
