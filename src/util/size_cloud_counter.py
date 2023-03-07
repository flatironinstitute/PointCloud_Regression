import regression.file_util as F
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description = 'read top folder, and count number of clouds inside')
    parser.add_argument('top_path', type = str, help = 'path of a category')

    args = parser.parse_args()

    top_folder = args.top_path
    file_list = F.list_files_in_dir(top_folder)

    sizes = []
    for f in file_list:
        curr_vert = F.read_off_file(f)
        sizes.append(len(curr_vert))
        if 40000 <= len(curr_vert) <= 45000:
            print("size of ", f[-8:-4], " is ", len(curr_vert))
    np.savetxt('cloud_size.txt',sizes)


if __name__ == '__main__':
    main()