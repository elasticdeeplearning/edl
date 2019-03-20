import os
import cPickle
import paddle
import glob

def prepare_dataset(output_path, name_prefix, reader_func, sample_count=128):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    suffix = "%s/%s-%%05d.pickle" % (output_path, name_prefix)
    lines = []
    indx_f = 0
    for i, d in enumerate(reader_func()):
        lines.append(d)
        if i >= sample_count and i % sample_count == 0:
            with open(suffix % indx_f, "w") as f:
                cPickle.dump(lines, f)
                lines = []
                indx_f += 1
    if lines:
        with open(suffix % indx_f, "w") as f:
            cPickle.dump(lines, f)

def cluster_reader(files_path, trainers, trainer_id):
    def reader():
        flist = glob.glob(files_path)
        flist.sort()
        my_file_list = []
        for idx, fn in enumerate(flist):
            if idx % trainers == trainer_id:
                print("append file for current trainer: %s" % fn)
                my_file_list.append(fn)

        for fn in my_file_list:
            print("processing file: ", fn)
            with open(fn, "r") as f:
                lines = cPickle.load(f)
                for line in lines:
                    yield line
    return reader