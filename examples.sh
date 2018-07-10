# python common/stitch.py -f rgbd_blender_data/0000.png -s rgbd_blender_data/0074.png -x 0 14 14 -y 0 4 4

python gammas.py --dir blender_data --rig blender_data/cameraSettings.json --rgb blender_data --depth blender_data --out blender_output --outfile blender_output/gammas.csv --method dibr
python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas.csv
python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas-mse-approx.csv --method mse-approx
python gammas.py --dir blender_data --rig blender_data/cameraSettings-vertical.json --out blender_output_vertical --outfile blender_output_vertical/gammas.csv

python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas.csv --method dsqm
python shortest_path.py --infile blender_output/gammas.csv
python solve2d.py --infile blender_output/gammas.csv --ys 8

