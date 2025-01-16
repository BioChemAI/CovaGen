import logging
import numpy as np

from utils.common import statistics_to_str

def analysis_source_and_vocab(source_list, vocab_dict):

    src_length_list = np.array([len(x["src"]) for x in source_list])
    trg_length_list = np.array([len(x["trg"]) for x in source_list if len(x["trg"])])

    split_list = [x["split"] for x in source_list]
    split_count = dict((k, split_list.count(k)) for k in ("train", "test", "valid", "other"))

    split_valid_trg_list = [x["split"] for x in source_list if len(x["trg"])]
    split_valid_trg_count = dict((k, split_valid_trg_list.count(k)) for k in ("train", "test", "valid", "other"))

    logging.info(f"len(source_list): {len(source_list)}")
    logging.info(f"split_count: {split_count}")
    logging.info(f"split_valid_trg_count: {split_valid_trg_count}")
    logging.info(statistics_to_str(src_length_list, "source_src"))
    logging.info(statistics_to_str(trg_length_list, "source_trg"))

    src_length_list = np.array([len(x) for x in vocab_dict["src"]])
    trg_length_list = np.array([len(x) for x in vocab_dict["trg"]])

    logging.info(f"len(vocab_dict['src']): {len(vocab_dict['src'])}, len(vocab_dict['trg']): {len(vocab_dict['trg'])}")
    logging.info(statistics_to_str(src_length_list, "vocab_src"))
    logging.info(statistics_to_str(trg_length_list, "vocab_trg"))
