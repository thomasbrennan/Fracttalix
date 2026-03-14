#!/bin/bash
# Run this once to set GitHub repo topics and description for SEO.
# Requires: gh auth login

gh repo edit thomasbrennan/Fracttalix \
  --description "Unified scientific corpus: Fractal Rhythm Model, Meta-Kaizen governance (8 papers), Dual Reader Standard (DRS), and DRS-MP — the first inter-AI communication protocol with epistemologically typed claims. 175+ machine-verifiable claims. Multi-AI hostile review relay across 9 providers. CC0 public domain."

gh repo edit thomasbrennan/Fracttalix --add-topic anomaly-detection
gh repo edit thomasbrennan/Fracttalix --add-topic streaming
gh repo edit thomasbrennan/Fracttalix --add-topic time-series
gh repo edit thomasbrennan/Fracttalix --add-topic fractal-rhythm-model
gh repo edit thomasbrennan/Fracttalix --add-topic dual-reader-standard
gh repo edit thomasbrennan/Fracttalix --add-topic inter-ai-communication
gh repo edit thomasbrennan/Fracttalix --add-topic multi-agent-systems
gh repo edit thomasbrennan/Fracttalix --add-topic machine-verifiable-claims
gh repo edit thomasbrennan/Fracttalix --add-topic falsification
gh repo edit thomasbrennan/Fracttalix --add-topic ai-peer-review
gh repo edit thomasbrennan/Fracttalix --add-topic drs-message-protocol
gh repo edit thomasbrennan/Fracttalix --add-topic canonical-build-plan
gh repo edit thomasbrennan/Fracttalix --add-topic meta-kaizen
gh repo edit thomasbrennan/Fracttalix --add-topic epistemology
gh repo edit thomasbrennan/Fracttalix --add-topic ai-governance
gh repo edit thomasbrennan/Fracttalix --add-topic agent-communication
gh repo edit thomasbrennan/Fracttalix --add-topic hostile-review
gh repo edit thomasbrennan/Fracttalix --add-topic python
gh repo edit thomasbrennan/Fracttalix --add-topic public-domain

echo "Done. Topics set."
