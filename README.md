To calculate the gammas (the dissimilarities between synthesized and original images), we can do so by running

```bash
python gammas.py --rig blender_data/cameraSettings.json --rgb blender_data --depth blender_data --out blender_output --outfile blender_output/gammas.csv --method dibr
```
The *rgb* and *depth* arguments hold the locations, where our rgb and depth data is stored.

Any images being synthesized in the process will be stored in the *out* directory, whereas the gamma values are stored in the *outfile* csv file.

The *method* argument is per default "dibr". Other methods are "optflow-depth", which calculates the optical flow between two original depth images, and "dsqm".

The *rig* argument specifies a file, that describes the camera's rig intrinsic and extrinsic parameters. This .json file looks like

```javascript
{
  "xs": [0,30],
  "ys": [0, 8],
  "img_file": "{1:04d}.png",
  "depth_file": "{1:04d}.exr",
  "kalibration": [
    [1750.0,0.0,800.0],
    [0.0,2333.333251953125,600.0],
    [0.0,0.0,1.0]
  ],
  "rotation": [
    [0.013961239717900753,0.9998440742492676,-0.010816766880452633],
    [0.09986663609743118,-0.012158012017607689,-0.9949265718460083],
    [-0.9949029088020325,0.01281021349132061,-0.10002076625823975]
  ],
  "translation": [12.710973739624023,-2.4600002765655518,2.9118027687072754],
  "translation_x": [0.0,0.20,0.0],
  "translation_y": [0.0,0.0,-0.11]
}
```

where *xs* holds the x coordinate of the first camera (usually 0) and the amount of cameras along the x-axis. Likewise *ys* is the y coordinate of the first camera and the amount of cameras along the y-axis.

*img_file* and *depth_file* give us the names of the image and depth files. *{1}* corresponds to x+y*xs and *{1:04d}* ensures that those numbers are trailed with zeroes so that they have at least 4 digits, e.g. 0000, 0001, ..., 9999.

*kalibration* contains the global calibration matrix, *rotation* the global rotation matrix and *translation* the global translation vector. The translation for a given camera at coordinates (x,y) is given by: translation + x*translation_x + y*translation*y

Since the rig file and the rgb and depth images are all stored in the same directory, we can use shorter version:

```bash
python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas.csv
```

In order to process those gamma values, e.g. in order to calculate the shortest paths described in the reference view selection chapter in the thesis, we can do

```bash
python shortest_path.py --infile blender_output/gammas.csv -y 0 -k 2 --gammas_exclude dsqm
```

with *infile* being the *outfile* from the previous executions of gammas.py. shortest_path.py can be used only for the one-dimensional case and only for the x-axis. So we have to specify the cameras' y coordinate in *y*.

The amount of reference views per view is given by *k*.

Furthermore, instead of trying to find the shortest paths for all gamma types in the *infile*, we can exclude certain types by specifying them in the *gammas_exclude* argument.

An approximative solution to the two-dimensional problem can be executed by

```bash
python solve2d.py --infile blender_output/gammas.csv --ys 8 --ym 3
```

where *ys* is the amount of cameras along the y-axis and *ym* is the final amount of reference views in a reference view column.