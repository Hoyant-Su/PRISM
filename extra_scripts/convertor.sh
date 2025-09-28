#!/bin/bash

proj_space="/inspire/hdd/project/continuinglearning/suhaoyang-240107100018/suhaoyang-240107100018/research/PRISM"
pdf_dir="${proj_space}/extra_scripts/assets_pdf"
svg_dir="${proj_space}/extra_scripts/assets_svg"
mkdir -p "$svg_dir"

for pdf_path in "$pdf_dir"/*.pdf
do
  [ -f "$pdf_path" ] || continue
  base_name=$(basename "$pdf_path" .pdf)
  svg_path="${svg_dir}/${base_name}.svg"
  echo "正在转换: $pdf_path -> $svg_path"
  pdf2svg "$pdf_path" "$svg_path"
done
