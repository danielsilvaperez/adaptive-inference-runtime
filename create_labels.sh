#!/bin/bash
# Script to create all necessary labels for the project
#
# Usage:
#   ./create_labels.sh

set -e

REPO="danielsilva010/Adaptive-Inference-Runtime"

echo "Creating GitHub labels for $REPO"
echo "=================================================="
echo ""

# Helper function to create a label
create_label() {
    local name="$1"
    local color="$2"
    local description="$3"

    if gh label create "$name" --repo "$REPO" --color "$color" --description "$description" 2>/dev/null; then
        echo "✓ Created label: $name"
    else
        echo "⚠ Label already exists or failed: $name"
    fi
}

# Phase labels
create_label "phase-0" "0E8A16" "Phase 0: Foundations"
create_label "phase-1" "1D76DB" "Phase 1: Small → Large Routing"
create_label "phase-2" "5319E7" "Phase 2: Speculative Decoding"
create_label "phase-3" "B60205" "Phase 3: KV Cache Compression"
create_label "phase-4" "D93F0B" "Phase 4: MacBook Enablement"
create_label "phase-5" "FBCA04" "Phase 5: Benchmarking + Proof"
create_label "phase-6" "0075CA" "Phase 6: Developer UX"

# Type labels
create_label "parallel" "C5DEF5" "Can be worked on in parallel"
create_label "sequential" "E99695" "Must be completed in order"

# Priority labels
create_label "P0" "B60205" "Critical priority"
create_label "P1" "D93F0B" "High priority"
create_label "P2" "FBCA04" "Medium priority"
create_label "P3" "C5DEF5" "Low priority"

# Technical area labels
create_label "infrastructure" "F9D0C4" "Infrastructure and setup"
create_label "documentation" "0075CA" "Documentation"
create_label "routing" "1D76DB" "Routing system"
create_label "confidence-scoring" "5319E7" "Confidence scoring metrics"
create_label "speculative-decoding" "D4C5F9" "Speculative decoding"
create_label "benchmarking" "FBCA04" "Benchmarking and evaluation"
create_label "kv-compression" "B60205" "KV cache compression"
create_label "safety" "D93F0B" "Safety and quality guards"
create_label "memory-management" "0E8A16" "Memory management"
create_label "macos" "E99695" "macOS/Metal specific"
create_label "visualization" "C5DEF5" "Runtime visualization"
create_label "developer-ux" "0075CA" "Developer experience"
create_label "configuration" "1D76DB" "Configuration system"
create_label "packaging" "F9D0C4" "Packaging and distribution"
create_label "publication" "FBCA04" "Publication and outreach"
create_label "optional" "C5DEF5" "Optional feature"

echo ""
echo "=================================================="
echo "Label creation complete!"
echo "=================================================="
