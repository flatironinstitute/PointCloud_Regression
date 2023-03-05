import regression.file_util as F
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description = 'read top folder, and count number of clouds inside')
    parser.add_argument('top_path', type = str, help = 'path of a category')

    args = parser.parse_args()

    top_folder = args.top_path
    file_list = F.list_files_in_dir(top_folder)

    min_size = float("inf")
    for f in file_list:
        curr_vert = F.read_off_file(f)
        print("size of cloud is: ", len(curr_vert))
        min_size = min(min_size, len(curr_vert))

    print("the min size of cloud is: ", min_size)

if __name__ == '__main__':
    main()