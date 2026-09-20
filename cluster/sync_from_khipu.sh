#!/bin/bash
# Pull results, logs and exported embeddings back into the repository.
set -e
REMOTE=${REMOTE:-ricardo.acuna@khipu.utec.edu.pe}
DEST=${DEST:-nlp-proyecto1}
rsync -avz "$REMOTE:$DEST/artifacts/results/" artifacts/results/
rsync -avz "$REMOTE:$DEST/artifacts/logs/" artifacts/logs/
rsync -avz "$REMOTE:$DEST/web/data/" web/data/
echo "results pulled from $REMOTE:$DEST"
