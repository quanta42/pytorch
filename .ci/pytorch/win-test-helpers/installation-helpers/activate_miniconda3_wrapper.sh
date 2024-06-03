#!/bin/bash

# Install Miniconda3
export INSTALLER_DIR="$SCRIPT_HELPERS_DIR"/installation-helpers

source "$INSTALLER_DIR"/activate_miniconda3.sh
env > tmp.txt
