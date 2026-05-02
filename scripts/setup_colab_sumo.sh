#!/usr/bin/env bash
set -euo pipefail

# Colab sometimes ships extra Launchpad PPAs that can be slow or unavailable.
# When apt tries to satisfy SUMO dependencies from those PPAs, SUMO installation
# can fail even though Ubuntu's standard repositories have usable packages.

echo "==> Disabling unstable Launchpad PPAs for this Colab runtime"
if [ -d /etc/apt/sources.list.d ]; then
  while IFS= read -r file; do
    if grep -Eq 'ppa.launchpadcontent.net/(ubuntugis|deadsnakes|graphics-drivers)' "$file"; then
      echo "Disabling PPA entries in $file"
      sudo sed -i.bak -E 's|^deb |# deb |g' "$file"
    fi
  done < <(find /etc/apt/sources.list.d -type f -name '*.list' -print)
fi

echo "==> Updating apt package index"
sudo apt-get clean
sudo apt-get update -o Acquire::Retries=3

echo "==> Installing SUMO"
sudo apt-get install -y --fix-missing sumo sumo-tools sumo-doc

echo "==> Verifying SUMO binary"
sumo --version

echo "==> Verifying project TraCI connectivity"
python scripts/check_sumo.py
