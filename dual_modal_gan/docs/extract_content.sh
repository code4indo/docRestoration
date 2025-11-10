#!/bin/bash

# Extract content dari chapter tanpa LaTeX header/footer

# Chapter 1
echo "Extracting chapter 1..."
tail -n +81 chapter1_pendahuluan.tex | head -n -1 > chapter1_pendahuluan_content.tex

# Chapter 2
echo "Extracting chapter 2..."
tail -n +75 chapter2_tinjauan_pustaka.tex | head -n -1 > chapter2_tinjauan_pustaka_content.tex

# Chapter 3
echo "Extracting chapter 3..."
tail -n +64 chapter3_metodologi.tex | head -n -1 > chapter3_metodologi_content.tex

# Chapter 4
echo "Extracting chapter 4..."
tail -n +59 chapter4_analysis_design.tex | head -n -1 > chapter4_analysis_design_content.tex

echo "Done! Created 4 content-only files."
