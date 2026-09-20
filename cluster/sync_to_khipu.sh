#!/bin/bash
# Push the code to the cluster. Results come back with sync_from_khipu.sh.
set -e
REMOTE=${REMOTE:-ricardo.acuna@khipu.utec.edu.pe}
DEST=${DEST:-nlp-proyecto1}
ssh "$REMOTE" "mkdir -p $DEST/artifacts/logs $DEST/artifacts/results $DEST/web/data"
rsync -avz --delete src/ "$REMOTE:$DEST/src/"
rsync -avz cluster/ "$REMOTE:$DEST/cluster/"
echo "code synced to $REMOTE:$DEST"
