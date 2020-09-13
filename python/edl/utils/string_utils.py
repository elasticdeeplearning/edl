
def dataset_to_string(o):
    """
    FileMeta to string
    """
    ret = "idx_in_list:{}, file_path:{}".format(o.idx_in_list, o.file_path)

    ret += " record:["
    for rs in o.records:
        for rec_no in range(rs.begin, rs.end + 1):
            ret += "(record_no:{})".format(rec_no)
    ret += "]"

    return ret


def data_request_to_string(o):
    """
    DataMeta to string
    """
    ret = "idx_in_list:{} file_path:{}".format(o.idx_in_list, o.file_path)
    for rs in o.chunks:
        ret += " chunk:["
        ret += chunk_to_string(rs)
        ret += "]"

    return ret


def chunk_to_string(rs):
    ret = "status:{} ".format(rs.status)
    for rec_no in range(rs.meta.begin, rs.meta.end + 1):
        ret += "(record_no:{}) ".format(rec_no)

    return ret

def bytes_to_string(o, codec='utf-8'):
    if o is None:
        return None

    if not isinstance(o, str):
        return o.decode(codec)

    return o