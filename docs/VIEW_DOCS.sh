#!/bin/bash

echo "======================================"
echo "  quant-gemm-from-scratch 文档列表"
echo "======================================"
echo ""

echo "📚 教程和指南 (docs/guides/):"
ls -lh docs/guides/*.md | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "📊 测试相关 (docs/testing/):"
ls -lh docs/testing/*.md | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "📈 报告 (docs/reports/):"
ls -lh docs/reports/*.md | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "🔬 分析 (docs/analysis/):"
ls -lh docs/analysis/*.md | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

echo "======================================"
echo "  最新添加 (2026-01-29)"
echo "======================================"
echo "  ⚡ CUDA-GEMM-BENCHMARK-TUTORIAL.md"
echo "  ⚡ QUICK-REFERENCE.md"
echo "  ⚡ CUDA-12.8-TEST-LOG.md"
echo "  ⚡ LLAMA_CPP_PERFORMANCE_COMPARISON.md"
echo "  ⚡ 2026-01-29-TEST-SUMMARY.md"
echo ""

echo "💡 快速访问:"
echo "  教程: docs/guides/CUDA-GEMM-BENCHMARK-TUTORIAL.md"
echo "  参考: docs/guides/QUICK-REFERENCE.md"
echo "  报告: docs/reports/LLAMA_CPP_PERFORMANCE_COMPARISON.md"
echo ""
