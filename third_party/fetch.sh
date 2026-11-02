#!/bin/bash
# fetch.sh -- clones the third-party code that code/ and eu/experiments/ import into ext/, at the exact commits we ran (unmodified).
#   usage (repository root): bash third_party/fetch.sh            (needs git and network access)
set -e
cd "$(dirname "$0")/.."
mkdir -p ext
clone() {   # clone <url> <folder> <commit>
  if [ ! -d "ext/$2/.git" ]; then git clone --quiet "$1" "ext/$2"; fi
  git -C "ext/$2" fetch --quiet origin
  git -C "ext/$2" checkout --quiet "$3"
  echo "ext/$2 @ $(git -C "ext/$2" rev-parse --short HEAD)"
}
clone https://github.com/NVlabs/edm.git                 edm                 008a4e5316c8e3bfe61a62f874bddba254295afb
clone https://github.com/NVlabs/edm2.git                edm2                4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc
clone https://github.com/openai/improved-diffusion.git  improved-diffusion  1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5
clone https://github.com/openai/guided-diffusion.git    guided-diffusion    22e0df8183507e13a7813f8d38d51b072ca1e67c
clone https://github.com/RouzAY/gepc-diffusion.git      gepc_repo           1b01cf26785dfa9418582fd0680fdbf02a29912e
