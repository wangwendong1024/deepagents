#!/usr/bin/env bash
set -euo pipefail

model_root=/srv/models
expected_uuid=e6f64cc3-6b39-46bf-9001-84ea946b07df

if [[ $(findmnt -rn -M "$model_root" -o UUID,FSTYPE) != "$expected_uuid xfs" ]]; then
    echo "The dedicated model filesystem is unavailable or has the wrong identity." >&2
    exit 1
fi

model_path=$(realpath -e -- "$1")
if [[ "$model_path" != "$model_root"/mydeepagent/huggingface/* ||
      ! -f "$model_path" || ! -r "$model_path" ||
      $(findmnt -rn -T "$model_path" -o TARGET,UUID) != "$model_root $expected_uuid" ]]; then
    echo "The configured model weight is not readable on the dedicated model filesystem." >&2
    exit 1
fi
