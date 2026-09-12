#!/usr/bin/env bash
# Creates a paid L4 VM. Never accepts a suffix or handles wallet result files.
set -euo pipefail

usage() {
  echo 'Usage: bash deploy/gcloud_deploy.sh PROJECT [ZONE] [VM_NAME]'
  echo 'Defaults: asia-southeast1-c, tron-vanity-l4. Creates paid resources.'
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ $# -lt 1 || $# -gt 3 ]]; then usage >&2; exit 2; fi
readonly PROJECT=$1
readonly ZONE=${2:-asia-southeast1-c}
readonly VM=${3:-tron-vanity-l4}
[[ $PROJECT =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || { echo 'Invalid project ID' >&2; exit 2; }
[[ $ZONE =~ ^[a-z]+-[a-z]+[0-9]+-[a-z]$ ]] || { echo 'Invalid zone' >&2; exit 2; }
[[ $VM =~ ^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$ ]] || { echo 'Invalid VM name' >&2; exit 2; }
command -v gcloud >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO_ROOT
cd "$REPO_ROOT"
git diff --quiet HEAD -- src deploy requirements.txt || {
  echo 'Commit deployment/source changes first; only committed files are deployed.' >&2; exit 1;
}
git cat-file -e HEAD:deploy/gcloud_deploy.sh
COMMIT=$(git rev-parse HEAD)
readonly COMMIT
EXISTING=$(gcloud compute instances list --project="$PROJECT" --zones="$ZONE" \
  --filter="name=$VM" --format='value(name)')
readonly EXISTING
[[ -z $EXISTING ]] || { echo "VM $VM already exists; see recovery instructions. Nothing changed." >&2; exit 1; }

PACKAGE_DIR=$(mktemp -d)
readonly PACKAGE_DIR
readonly PACKAGE="$PACKAGE_DIR/tron-source.tar.gz"
readonly DEPLOY_TOKEN=${PACKAGE_DIR##*/}
# Archive only committed application/deployment files, never home/results or old/.
git archive --format=tar.gz --output="$PACKAGE" HEAD src deploy requirements.txt
owned_vm=false
ready=false
finish() {
  local status=$?
  trap - EXIT
  if [[ $owned_vm == true ]]; then
    if ! gcloud compute instances delete-access-config "$VM" --project="$PROJECT" \
      --zone="$ZONE" --access-config-name='External NAT' --quiet; then
      echo 'WARNING: Could not remove the temporary public IP. Check the VM manually.' >&2
      status=1
    fi
    if [[ $ready != true || $status != 0 ]]; then
      echo 'Deployment incomplete; stopping the VM. Disk is retained and still billed.' >&2
      gcloud compute instances stop "$VM" --project="$PROJECT" --zone="$ZONE" --quiet || \
        echo 'WARNING: Stop failed; check the VM now to avoid ongoing GPU charges.' >&2
      status=1
    else
      echo "DEPLOYMENT_READY=$COMMIT"
      echo "SSH: gcloud compute ssh $VM --project=$PROJECT --zone=$ZONE --tunnel-through-iap"
    fi
  fi
  # Only remove the exact temporary archive we created, not any recursive directory.
  rm -f "$PACKAGE"
  rmdir "$PACKAGE_DIR"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

FIREWALL=$(gcloud compute firewall-rules list --project="$PROJECT" \
  --filter='name=allow-iap-ssh-tron-vanity' --format='value(name)')
readonly FIREWALL
if [[ -z $FIREWALL ]]; then
  gcloud compute firewall-rules create allow-iap-ssh-tron-vanity --project="$PROJECT" \
    --network=default --direction=INGRESS --action=ALLOW --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 --target-tags=iap-ssh --priority=900 --quiet
fi

echo "Creating paid L4: $PROJECT / $ZONE / $VM; commit $COMMIT"
# Temporary external IP supplies outbound access to apt/PyPI. Removed by finish().
if ! gcloud compute instances create "$VM" --project="$PROJECT" --zone="$ZONE" \
  --machine-type=g2-standard-4 --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE --provisioning-model=STANDARD \
  --network=default --subnet=default --tags=iap-ssh \
  --metadata="enable-oslogin=TRUE,tron-deploy-commit=$COMMIT,tron-deploy-token=$DEPLOY_TOKEN" \
  --labels=purpose=tron-vanity --no-service-account --no-scopes \
  --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --quiet; then
  # An API timeout can happen after creation. Only touch a VM carrying this run's token.
  if TOKEN=$(gcloud compute instances describe "$VM" --project="$PROJECT" --zone="$ZONE" \
    --format='json(metadata.items)' | python3 -c \
    'import json,sys; print(next((i["value"] for i in json.load(sys.stdin).get("metadata", {}).get("items", []) if i["key"] == "tron-deploy-token"), ""))') && [[ $TOKEN == "$DEPLOY_TOKEN" ]]; then
    owned_vm=true
  else
    echo 'Create failed; ownership could not be confirmed. Check cloud resources manually.' >&2
  fi
  exit 1
fi
owned_vm=true

remote() {
  gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" \
    --tunnel-through-iap --quiet --command="$1" \
    --ssh-flag='-o ConnectTimeout=10' --ssh-flag='-o ConnectionAttempts=1'
}
wait_for_ssh() {
  local attempt
  for attempt in {1..40}; do
    if remote 'true' >/dev/null 2>&1; then return 0; fi
    sleep 10
  done
  echo 'SSH unavailable: check OS Login admin / IAP permissions and firewall.' >&2
  return 1
}
wait_for_ssh
gcloud compute scp "$PACKAGE" "$VM:tron-source-$COMMIT.tar.gz" \
  --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap --quiet
remote "sudo install -d -m 0755 /opt/tron-vanity && sudo tar -xzf tron-source-$COMMIT.tar.gz -C /opt/tron-vanity && echo $COMMIT | sudo tee /opt/tron-vanity/.deployed-commit >/dev/null"
remote 'sudo bash /opt/tron-vanity/deploy/install_gpu_driver.sh'
if ! remote 'nvidia-smi >/dev/null'; then
  # Schedule reboot after SSH has returned, so a disconnect is not a fake success.
  remote "sudo shutdown -r +1 'Complete NVIDIA driver setup'"
  sleep 45
  sleep 45
  wait_for_ssh
  remote 'nvidia-smi >/dev/null'
fi
remote 'sudo bash /opt/tron-vanity/deploy/install_app.sh'
# Runs as the SSH user, warming that user's CuPy cache. Never prints or saves keys.
remote 'PYTHONPATH=/opt/tron-vanity/src /opt/tron-vanity/.venv/bin/python /opt/tron-vanity/deploy/verify_gpu.py && tron-vanity --help'
ready=true
