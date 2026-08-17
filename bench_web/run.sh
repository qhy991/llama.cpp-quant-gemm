#!/bin/bash
# Run Quant-GEMM Kernel Bench Web UI

cd "$(dirname "$0")"

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null

# Activate environment
conda activate KM-12.8

echo "🚀 Starting Quant-GEMM Kernel Bench..."
echo "   URL: http://localhost:8511"
echo ""

streamlit run app.py --server.port 8511 --server.headless true
