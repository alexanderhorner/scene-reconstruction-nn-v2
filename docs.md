#Unet


# Helpful notes and snippets

conda enviroment name: tf-latest

Conda cheat sheet: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

apple tensorflow plugin instalation: https://developer.apple.com/metal/tensorflow-plugin/

# Convert training pporgress results to video
ffmpeg -framerate 30 -pattern_type glob -i '*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4

ffmpeg -framerate 10 -pattern_type glob -i '*.png' -c:a copy -shortest -c:v libx264 -pix_fmt yuv420p ../out.mp4


## delete .ds-store in subfolder (terminal)
find . -name ".DS_Store" -type f -delete -print

# run info
## simplecars-norotation-fixedpos-whitebg-squarelines 20221105-141208-PAPER-FIRST
Batch size 4
MEGA gutes ergebnis fist try