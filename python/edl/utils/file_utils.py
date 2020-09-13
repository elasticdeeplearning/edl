def read_txt_lines(file_list):
    """
    return [(file_path, line_no)...]
    """
    line_no = -1
    ret = []
    with open(file_list, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue

            line_no += 1
            ret.append((line, line_no))
    return ret