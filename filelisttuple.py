import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data', type=str,
                    help='The folder path')
parser.add_argument('--train_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The train filename.')
parser.add_argument('--validation_filename', default='./data_flist/validation_shuffled.flist', type=str,
                    help='The validation filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to be shuffled')
parser.add_argument('--label_contains', default='catIds', type=str,
                    help='Needed to be shuffled')
parser.add_argument('--label_folder_path', default='./training_data', type=str,
                    help='The folder path')

if __name__ == "__main__":

    args = parser.parse_args()

    # get the list of directories and separate them into 2 types: training and validation
    training_dirs = os.path.join(args.folder_path , "training")
    training_label_dirs = os.path.join(args.label_folder_path ,"training")
    validation_dirs = os.path.join(args.folder_path, "validation")
    validation_label_dirs = os.path.join(args.label_folder_path, "validation")

    # make 2 lists to save file paths
    training_file_names = []
    training_file_label_names = []
    validation_file_names = []
    validation_file_label_names = []
    """
    # append all files into 2 lists
    for training_dir in training_dirs:
        # append each file into the list file names
        training_folder = os.listdir(args.folder_path + "/training" + "/" + training_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            training_item = args.folder_path + "/training" + "/" + training_dir + "/" + training_item
            training_file_names.append(training_item)
    """

    for root, dirs, files in os.walk(training_dirs, topdown=False):
        for name in files:
            training_file_names.append(os.path.join(root,name))

    for root, dirs, files in os.walk(validation_dirs, topdown=False):
        for name in files:
            validation_file_names.append(os.path.join(root,name))


    for root, dirs, files in os.walk(training_label_dirs, topdown=False):
        for name in files:
            if 'catIds' in name:
                training_file_label_names.append(os.path.join(root,name))

    for root, dirs, files in os.walk(validation_label_dirs, topdown=False):
        for name in files:
            if 'catIds' in name:
                validation_file_label_names.append(os.path.join(root,name))

    """
    # append all files into 2 lists
    for validation_dir in validation_dirs:
        # append each file into the list file names
        validation_folder = os.listdir(args.folder_path + "/validation" + "/" + validation_dir)
        for validation_item in validation_folder:
            # modify to full path -> directory
            validation_item = args.folder_path + "/validation" + "/" + validation_dir + "/" + validation_item
            validation_file_names.append(validation_item)

    # append all files into 2 lists
    for training_label_dir in training_label_dirs:
        # append each file into the list file names
        training_folder = os.listdir(args.label_folder_path + "/training" + "/" + training_label_dir)
        for training_item in training_folder:
            # modify to full path -> directory
            if args.label_contains in training_item:
                training_item = args.label_folder_path + "/training" + "/" + training_dir + "/" + training_item
                training_file_label_names.append(training_item)

    # append all files into 2 lists
    for validation_label_dir in validation_label_dirs:
        # append each file into the list file names
        validation_folder = os.listdir(args.label_folder_path + "/validation" + "/" + validation_label_dir)
        for validation_item in validation_folder:
            # modify to full path -> directory
            if args.label_contains in validation_item:
                validation_item = args.label_folder_path + "/validation" + "/" + validation_dir + "/" + validation_item
                validation_file_label_names.append(validation_item)
    """
    print(len(training_file_names))
    print(len(training_file_label_names))
    training_file_names.sort()
    training_file_label_names.sort()
    validation_file_names.sort()
    validation_file_label_names.sort()
    training_tuple_list = []
    validation_tuple_list = []
    for idx in range(len(training_file_names)):
        training_tuple_list.append((training_file_names[idx],training_file_label_names[idx] ))
    for idx in range(len(validation_file_names)):
        validation_tuple_list.append((validation_file_names[idx],validation_file_label_names[idx] ))

    # print all file paths
    for i in training_tuple_list:
        print(i)
    for i in validation_tuple_list:
        print(i)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(training_tuple_list)
        shuffle(validation_tuple_list)

    # make output file if not existed
    if not os.path.exists(args.train_filename):
        os.mknod(args.train_filename)

    if not os.path.exists(args.validation_filename):
        os.mknod(args.validation_filename)
    training_string_list = []
    for i in training_tuple_list:
        training_string_list.append(i[0] + ';' + i [1])
    validation_string_list = []
    for i in validation_tuple_list:
        validation_string_list.append(i[0] + ';' + i [1])
    # write to file
    fo = open(args.train_filename, "w")
    fo.write("\n".join(training_string_list))
    fo.close()

    fo = open(args.validation_filename, "w")
    fo.write("\n".join(validation_string_list))
    fo.close()

    # print process
    print("Written file is: ", args.train_filename, ", is_shuffle: ", args.is_shuffled)