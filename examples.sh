# python common/stitch.py -f rgbd_blender_data/0000.png -s rgbd_blender_data/0074.png -x 0 14 14 -y 0 4 4
python gammas.py --dir rgbd_data --out rgbd_output --outfile rgbd_output/gammas.csv
python shortest_path.py --infile rgbd_output/gammas.csv

python gammas.py --dir kitchen_data --out kitchen_output --outfile kitchen_output/gammas.csv
python shortest_path.py --infile kitchen_output/gammas.csv

python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas.csv --append
python gammas.py --dir blender_data --out blender_output --outfile blender_output/gammas2.csv --method dsqm
python shortest_path.py --infile blender_output/gammas.csv



from yomi_base.anki_bridge import yomichanInstance
def prnt(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    sys.stderr.write(str(exc_tb.tb_lineno))
    sys.stderr.write(e.__str__())
print(yomichanInstance.loadLanguage("spanish",prnt))