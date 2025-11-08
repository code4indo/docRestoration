#!/bin/bash

# Fix LaTeX errors in jatniko_id.tex

FILE="jatniko_id.tex"

# Replace double backslashes with single backslashes for lambda
sed -i 's/\\\\lambda_{\\\\text{/$\\\\lambda_{/g' "$FILE"
sed -i 's/\\\\lambda_{\\\\text{/$\\\\lambda_{/g' "$FILE"

# Fix percent signs in tables
sed -i 's/CER (\\\\%)/CER (\%)}/g' "$FILE"
sed -i 's/WER (\\\\%)/WER (\%)}/g' "$FILE"

# Fix math mode issues
sed -i 's/CER (\\\\%) \\$ \\\\downarrow\\$}/CER (\%)}/g' "$FILE"
sed -i 's/WER (\\\\%) \\$ \\\\downarrow\\$}/WER (\%)}/g' "$FILE"

# Fix double backslashes in loss function notation
sed -i 's/\\\\mathcal{L}/\\$ \\mathcal{L}/g' "$FILE"

# Fix fbox parbox newlines
sed -i 's/}\\\\textbf{/}\\\\\n\\textbf{/g' "$FILE"

echo "Fixed LaTeX errors in $FILE"