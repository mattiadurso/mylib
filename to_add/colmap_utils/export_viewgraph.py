import os
import glob
from tqdm.auto import tqdm
from pathlib import Path


def export_viewgraph(db_path: str, min_matches: int, db_name: str = "database.db", script="/home/mattia/Desktop/Repos/colmap/scripts/python/export_inlier_pairs.py"):
    """
    db_path: ds path with regex to match all the scenes
    min_matches: minimum number of matches to consider a pair of images
    db_name: name of the database file
    """
    for scene_path in tqdm(sorted(glob.glob(db_path))):
        print(Path(scene_path, db_name))
        os.system(f"python {script} \
                --database_path {Path(scene_path, db_name)} \
                --match_list_path {Path(scene_path)}/viewgraph_{min_matches}.txt \
                --min_num_matches {min_matches}")



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Export viewgraph from COLMAP database')
    parser.add_argument('--db_path', type=str, help='Path to the database')
    parser.add_argument('--min_matches', type=int, help='Minimum number of matches to consider a pair of images')
    args = parser.parse_args()

    export_viewgraph(args.db_path, args.min_matches)


# Example usage
'''
python mylib/colmap_utils/export_viewgraph.py --db_path "/home/mattia/Desktop/datasets/Megadepth/data/scenes/*" --min_matches 50

python mylib/colmap_utils/export_viewgraph.py --db_path "//home/mattia/Desktop/datasets/IMB/phototourism/*" --min_matches 50


'''
