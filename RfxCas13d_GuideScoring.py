import sys
import subprocess
import string
import re
import random
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestRegressor
from optparse import OptionParser
import argparse
from math import log10
from datetime import datetime
from Bio.SeqIO import DNAStringSet
from Bio.Seq import Seq
from Bio.Seq import reverse_complement
from Bio.Alphabet import generic_dna
from Bio import SeqIO


def get_options(logger: logging.Logger = None) -> argparse.Namespace:
    """
    Parses command line options for the Cas13 sgRNA Design Predictor.
    
    :return: An object containing all the parsed command line options.
    """
    parser = argparse.ArgumentParser(
        description="Cas13 sgRNA Design Predictor",
        epilog="""
        Example: 
            python Cas13designGuidePredictor.py -i input.fa -o output \\
                -m Cas13designGuidePredictorInput.csv -p True -t 5 -l 25 \\
                -g 0.1-0.9 --homopolymer_length 3,4 --window_offset 0 \\
                --direct_repeat aacccctaccaactggtcggggtttgaaac \\
                --rna_fold_path ./scripts/ --rna_fold ./scripts/ \\
                --log Cas13designGuidePredictor --log_level INFO 
        Short:
            python Cas13designGuidePredictor.py -i input.fa -o output \\
                """)
    parser.add_argument(
        '-i', '--input', dest='input', type=str, metavar='String', required=True,
        default=False, help='Input fasta file, split by ",".')
    parser.add_argument(
        '-o', '--output', dest='output', type=str, metavar='Dir', required=True,
        default='./', help='output dir. Default: ./')
    parser.add_argument(
        '-m', '--model', dest='rfmodel', type=str, metavar='String', required=True,
        default='/data/biosoft/cas13design/Cas13design/data/Cas13designGuidePredictorInput.csv', 
        help='Random Forest Model input <Cas13designGuidePredictorInput.csv>')
    parser.add_argument(
        '-p', '--plot', dest='plot', type=bool, metavar='Bool', 
        default=True, help='if you would like the results plotted <True> or <False>')
    parser.add_argument(
        '-t', '--threads', dest='threads', type=int, metavar='Int', 
        default=5, help='threads number. Default: 5')
    parser.add_argument(
        '-l', '--length', dest='length', type=int, metavar='Int', 
        default=25, help='Guide sgRNA length. Default: 25')
    parser.add_argument(
        '-g', '--gc_range', dest='gc_range', type=str, metavar='String', 
        default='0.1-0.9', help='GC content range. Default: 0.1-0.9')
    parser.add_argument(
        '--homopolymer_length', dest='homopolymer_length', type=str, metavar='String', 
        default='3,4', help='Homopolymer T and nonT length. Default: 3,4')
    parser.add_argument(
        '--direct_repeat', dest='direct_repeat', type=str, metavar='String', 
        default='aacccctaccaactggtcggggtttgaaac', help='Direct repeat sequence. Default: aacccctaccaactggtcggggtttgaaac')
    parser.add_argument(
        '--rna_fold_path', dest='rna_fold_path', type=str, metavar='String', 
        default='./scripts/', help='RNAfold path. Default: ./scripts/')
    parser.add_argument(
        '--window_offset', dest='window_offset', type=int, metavar='Int', 
        default=0, help='Offset to start scoring guides. Some of the features will depend of the target sequence context upstream. Without the offset, NAs will be assigned for the respective feature. Default: 0')
    parser.add_argument(
        '--log', dest='log', type=str, metavar='String', 
        default='Cas13designGuidePredictor', help='log file name. Default: Cas13designGuidePredictor')
    parser.add_argument(
        '--log_level', dest='log_level', type=str, metavar='String', 
        default='INFO', help='log level. Default: DEBUG')
    parser.add_argument(
        '--rna_fold', dest='rna_fold', type=str, metavar='String', 
        default=False, help='RNAfold path. Default: False')
    # Add additional information to help message
    parser._positionals.title = 'Positional arguments'
    parser._optionals.title = 'Optional arguments'
    options = parser.parse_args()
    logger.info('Parsed command line options:')
    logger.info(options)
    if not options.input or not options.rfmodel:
        parser.print_help()
        logger.error(f'Wrong input options, please check your input!')
        sys.exit()
    options.gc_range = [float(i.strip()) for i in options.gc_range.split('-')]
    options.homopolymer_length = [int(i.strip()) for i in options.homopolymer_length.split(',')]
    options.log_level = logging.getLevelName(options.log_level)
    return options


def set_logger(name: str, level: str='DEBUG') -> logging.Logger:
    """
    Creates and configures a new logger with the specified name and level.

    Args:
        name (str): The name of the logger.
        level (int): The log level to set. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: The newly created and configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    basic_format = '%(asctime)s :: %(name)s :: %(levelname)s :: %(funcName)s :: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(basic_format, date_format)
    # 输出到控制台
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level=level)
    # 输出到日志文件
    log_file = f'{name}.log.txt'
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(level=level)
    handler.setFormatter(formatter)
    # 添加
    logger.addHandler(console)
    logger.addHandler(handler)
    return logger


def get_executable_path(executable: str) -> str:    
    """
    Given the name of an executable, this function returns the path to the executable if it exists in the system's PATH variable, otherwise it returns None.

    :param executable: A string representing the name of the executable.
    :type executable: str
    :return: The path to the executable if it exists in the system's PATH variable. Otherwise None.
    :rtype: str or None
    """
    
    result = subprocess.run(["which", executable], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None
    

def give_rna_fold_path(rna_fold_path: str, logger: logging.Logger = None) -> dict:
    """
    Returns the RNAfold, RNAplfold, and RNAhybrid paths given the path to the RNAfold executable.

    :param rna_fold_path: A string representing the path to the RNAfold executable.
    :param logger: A logging.Logger object used to log information and errors.
    :return: A dictionary containing the RNAfold, RNAplfold, and RNAhybrid paths.
    """
    if not isinstance(rna_fold_path, str):
        logger.error(f'Wrong input options type, cannot give RNAfold path!')
        sys.exit()
    if rna_fold_path == 'local':
        rna_fold = get_executable_path('RNAfold')
        rna_plfold = get_executable_path('RNAplfold')
        rna_hybrid = get_executable_path('RNAhybrid')
        rna_hyb = None
    elif os.path.exists(rna_fold_path):
        rna_fold = os.path.join(rna_fold_path, "RNAfold")
        rna_plfold = os.path.join(rna_fold_path, "RNAplfold")
        rna_hyb = os.path.join(rna_fold_path, "RNAhyb.sh")
        rna_hybrid = None
    # 检查路径是否存在
    if os.path.exists(rna_fold) and os.path.exists(rna_plfold) and (os.path.exists(rna_hyb) or os.path.exists(rna_hybrid)):
        logger.info(f'RNAfold path: {rna_fold_path}')
        return {'rna_fold': rna_fold, 'rna_plfold': rna_plfold, 'rna_hyb': rna_hyb}
    else:
        logger.error(f'RNAfold path: {rna_fold_path} not exists!')
        sys.exit()
        
        
def load_sequences_from_fasta(fa_path: str, out_fa: str = 'merged.fasta', logger: logging.Logger = None) -> list:
    """
    Load sequences from a FASTA file and merge them into one sequence record.

    Args:
        fa_path (str): The path to the FASTA file.
        out_fa (str): The path to the output FASTA file. Default is 'merged.fasta'.
        logger (logging.Logger): The logger to record log messages.

    Returns:
        list: A list of sequence records.
    """
    # Check if multiple files
    if isinstance(fa_path, str) and ',' in fa_path:
        fa_path = [file.strip() for file in fa_path.split(',')]
    # Load sequences from each file
    merged_records = []
    for file in fa_path:
        # Check if file exists
        if os.path.exists(file):
            with open(file, 'r') as f:
                records = list(SeqIO.parse(f, 'fasta'))
                merged_records.extend(records)
        else:
            logger.warning(f'File {file} not exists!')
    # Merge all records into one sequence record
    merged_sequence = "".join([record.seq for record in merged_records])
    merged_record = SeqIO.SeqRecord(
        seq=merged_sequence,
        id=merged_records[0].id,
        description=merged_records[0].description)
    # Write merged record to file
    with open(out_fa, 'w') as f:
        SeqIO.write(merged_record, f, 'fasta')
    # Load sequences from output file
    records = list(SeqIO.parse(out_fa, 'fasta'))
    if logger:
        logger.info(f'Loaded {len(records)} sequences from {len(fa_path)} files.')
    return records


def filter_sequences_by_length(records: list, logger: logging.Logger) -> list:
    """
    Filter sequences by length and log messages for each sequence.

    Args:
        records (list): A list of sequence records.
        logger (logging.Logger): The logger to record log messages.

    Returns:
        list: A filtered list of sequence records.
    """
    filtered_records = []
    for record in records:
        length = len(record.seq)
        if length < 80:
            logger.warning(f'Sequence {record.id} length {length} is less than 80nt! RNAplfold requires a minimum of 80nt length for the target site accessibility calculation.')
        elif length < 30:
            logger.warning(f'Sequence {record.id} length {length} is less than 30nt! Input sequence too short. Please provide an input sequence >30nt to be able to assess the target site nucleotide context.')
        else:
            filtered_records.append(record)
            logger.info(f'Sequence {record.id} length {length} is OK.')
    return filtered_records


def get_gc_content(seq: str) -> float:
    """
    Calculate the GC content of a DNA sequence.

    Args:
        seq (str): A DNA sequence.

    Returns:
        float: The GC content of the sequence.
    """
    gc_count = seq.count("C") + seq.count("G")
    return gc_count / len(seq)


def get_matching_start_position(y: str, end: str = "end") -> int:
    """
    Extract the matching start position from a string.

    Args:
        y (str): A string in the format of "chr:start-end_strand".
        end (str): The end of the position to extract. Default is "end".

    Returns:
        int: The matching start position.
    """
    coord = y.split(":")[1].split("_")[0]
    if end == "end":
        return int(coord.split("-")[1])
    else:
        return int(coord.split("-")[0])
    
    
def get_longest_consecutive_bases(seq: str, base: str) -> int:
    """
    Get the length of the longest consecutive bases in a DNA sequence.

    Args:
        seq (str): A DNA sequence.
        base (str): The base to count.

    Returns:
        int: The length of the longest consecutive bases.
    """
    max_count = None
    count = None
    for i in seq:
        if i == base:
            if count is None:
                count = 1
            else:
                count += 1
            if max_count is None or count > max_count:
                max_count = count
        else:
            count = None
    return max_count if max_count is not None else 0


def get_all_consecutive_bases(seq: str) -> dict:
    """
    Get the length of the longest consecutive bases for each DNA base.

    Args:
        seq (str): A DNA sequence.

    Returns:
        dict: A dictionary of the longest consecutive bases for each DNA base.
    """
    consecutive_bases = {
        "A": get_longest_consecutive_bases(seq, "A"),
        "C": get_longest_consecutive_bases(seq, "C"),
        "G": get_longest_consecutive_bases(seq, "G"),
        "T": get_longest_consecutive_bases(seq, "T")
    }
    return consecutive_bases


def get_all_possible_guides(
    seq: str, 
    guide_length: int = 25, 
    gc_min: float = 0.2, 
    gc_max: float = 0.8, 
    homopolymer_length_t: int = 3, 
    homopolymer_length_nont: int = 4, 
    logger: logging.Logger = None) -> list:
    """
    Get all possible guide sequences in a DNA sequence.

    Args:
        seq (str): A DNA sequence.
        guide_length (int): The length of the guide sequence. Default is 25.
        gc_min (float): The minimum GC content of the guide sequence. Default is 0.1.
        gc_max (float): The maximum GC content of the guide sequence. Default is 0.9.
        homopolymer_length_t (int): The maximum length of a homopolymer of T. Default is 3.
        homopolymer_length_nont (int): The maximum length of a homopolymer of non-T bases. Default is 4.

    Returns:
        list: A list of all possible guide sequences.
    """
    # 获取所有可能的guide起始位置
    starts = range(1, len(seq) - guide_length + 2)
    # 获取所有可能的guide终止位置
    ends = range(guide_length, len(seq) + 1)
    # 获取所有可能的guide序列
    substrings = [seq[start-1:end] for start, end in zip(starts, ends)]
    # 获取guide的GC含量
    gc = [get_gc_content(s) for s in substrings]
    # 获取同源碱基的长度
    consecutive_bases = pd.DataFrame([get_all_consecutive_bases(s) for s in substrings])

    # 过滤
    gc_index = np.where((np.array(gc) > gc_min) & (np.array(gc) < gc_max))[0]
    consecutive_bases_index = np.where((consecutive_bases[["T", "C", "G"]].max(axis=1) <= homopolymer_length_nont) & (consecutive_bases[["A"]].max(axis=1) <= homopolymer_length_t))[0]

    if (len(gc_index) > 0) & (len(consecutive_bases_index) > 0):
        idx = np.intersect1d(gc_index, consecutive_bases_index)
        if len(idx) > 0:
            return [substrings[i] for i in idx]
        else:
            logger.warning(f'No intersect: GC and Homopolymers out of range!')
            return None
    else:
        if (len(gc_index) == 0) & (len(consecutive_bases_index) > 0):
            logger.warning(f'GC out of range!')
            return None
        elif (len(gc_index) > 0) & (len(consecutive_bases_index) == 0):
            logger.warning(f'Homopolymers out of range!')
            return None
        else:
            logger.warning(f'GC and Homopolymers out of range!')
            return None
        
        
def get_nt_density_vector(seq: Seq, nt: str = "G", window: int = 30) -> np.ndarray:
    """
    Get the density vector of a specific nucleotide in a DNA sequence.

    Args:
        seq (Seq): A DNA sequence.
        nt (str): The nucleotide for which to calculate the density. Default is "G".
        window (int): The size of the window used to calculate the density. Default is 30.

    Returns:
        np.ndarray: A density vector of the specified nucleotide.
    """
    d = window
    ma = np.full(len(seq), np.nan)
    nt_counts = np.array([1 if base == nt else 0 for base in seq])
    nt_counts = np.concatenate((np.zeros(d//2), nt_counts, np.zeros(d//2)))
    nt_counts_window = np.array([np.sum(nt_counts[i:i+d]) for i in range(len(nt_counts)-d+1)])
    ma[d//2:-d//2+1] = nt_counts_window / d
    return ma


def get_nt_densities(seq: Seq) -> dict:
    """
    Get the maximum and minimum density vectors of different nucleotides in a DNA sequence.

    Args:
        seq (Seq): A DNA sequence.

    Returns:
        dict: A dictionary containing the maximum and minimum density vectors of different nucleotides.
    """
    # Define the nucleotide types and window sizes to be calculated
    nts = {"A": None, "C": None, "G": None, "T": None, "AT": None, "GC": None}
    for nt in nts:
        nts[nt] = max(max.loc[max["NT"] == nt, "W"]), min(min.loc[min["NT"] == nt, "W"])
    # Calculate the density vectors of each nucleotide type in the maximum and minimum windows
    densities = {}
    for nt, (max_n, min_n) in nts.items():
        densities[f"max_{nt}"] = get_nt_density_vector(seq=seq, nt=nt, window=max_n)
        densities[f"min_{nt}"] = get_nt_density_vector(seq=seq, nt=nt, window=min_n)
    return densities


def get_value_at_position(pos: int, vec: np.ndarray, point: int = -11) -> np.ndarray:
    """
    Get the value at a specific position in a vector.

    Args:
        pos (int): The position of the value to get.
        vec (np.ndarray): The vector to get the value from.
        point (int): An offset used to calculate the index of the value. Default is -11.

    Returns:
        np.ndarray: The value at the specified position in the vector.
    """
    if np.isnan(pos):
        return np.full(len(vec), np.nan)
    else:
        idx = pos + point
        idx[idx < 0] = np.nan
        return vec[idx]
    
    
def get_nt_point_densities(dat: pd.DataFrame, seq: Seq, window_size: int = 30) -> pd.DataFrame:
    """
    Get the maximum and minimum point densities of different nucleotides at each position in a dataframe.

    Args:
        dat (pd.DataFrame): A dataframe containing match positions.
        seq (Seq): A DNA sequence.
        window_size (int): The size of the window used to calculate the densities. Default is 30.

    Returns:
        pd.DataFrame: A dataframe containing the maximum and minimum point densities of different nucleotides at each position.
    """
    # Define the nucleotide types and window sizes to be calculated
    nts = {"A": None, "C": None, "G": None, "T": None, "AT": None, "GC": None}
    for nt in nts:
        nts[nt] = max(dat.max().loc[dat.max()["NT"] == nt, "P"]), min(dat.min().loc[dat.min()["NT"] == nt, "P"])
    # Calculate the density vectors of each nucleotide type in the maximum and minimum windows
    densities = {}
    for nt, (max_p, min_p) in nts.items():
        densities[f"max_{nt}"] = get_nt_density_vector(seq=seq, nt=nt, window=window_size)
        densities[f"min_{nt}"] = get_nt_density_vector(seq=seq, nt=nt, window=window_size)
    # Calculate the point densities of each nucleotide type at each match position
    point_densities = []
    for nt, (max_p, min_p) in nts.items():
        point_densities.append(
            dat["MatchPos"].apply(
                get_value_at_position, vec=densities[f"max_{nt}"], point=max_p
            ).rename(f"NTdens_max_{nt}")
        )
        point_densities.append(
            dat["MatchPos"].apply(
                get_value_at_position, vec=densities[f"min_{nt}"], point=min_p
            ).rename(f"NTdens_min_{nt}")
        )
    dens = pd.concat(point_densities, axis=1)
    dens.index = dat.index
    out = pd.concat([dat, dens], axis=1)
    return out


def predict_guide_scores_with_model(
    data: pd.DataFrame, 
    fields: list, 
    model: RandomForestRegressor, 
    mean: np.ndarray, 
    std: np.ndarray) -> np.ndarray:
    """
    Use a trained random forest regression model to predict guide scores.

    Args:
        data (pd.DataFrame): A dataframe containing the input data.
        fields (list): A list of column names to include in the input data.
        model: The trained random forest regression model.
        mean: The mean of the input data used during training.
        std: The standard deviation of the input data used during training.

    Returns:
        np.ndarray: An array containing the predicted guide scores.
    """
    # Remove incomplete entries
    x = data.dropna(subset=fields)
    # Scale numeric values for prediction
    numeric = x.select_dtypes(include=float).columns.difference(["normCS", "Gquad", "DR", *[f"{nt}_" for nt in "ACGT"]])
    x.loc[:, numeric] = (x.loc[:, numeric] - mean) / std
    x.loc[:, numeric] = x.loc[:, numeric].clip(0, 1)
    # Predict guide scores using the trained model
    return model.predict(x.iloc[:, 1:])


def normalize_score(score: float, minimum: float, maximum: float) -> float:
    """
    Normalize a score using the minimum and maximum values.

    Args:
        score (float): The score to be normalized.
        minimum (float): The minimum value of the score.
        maximum (float): The maximum value of the score.

    Returns:
        float: The normalized score.
    """
    if np.isnan(score):
        return np.nan
    else:
        return (score - minimum) / (maximum - minimum)
    
    
def add_scores_to_dataframe(df: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """
    Add the given scores to the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add the scores to.
        scores (dict): A dictionary containing the scores to add.

    Returns:
        pd.DataFrame: The dataframe with the added scores.
    """
    # Match by guide name
    idx = df.index.intersection(scores.keys())
    if set(scores.keys()) == set(idx):
        # Initiate GuideScore column
        df["GuideScores"] = np.nan
        # Add scores given the positional index
        df.loc[idx, "GuideScores"] = list(scores.values())
        # Initiate rank column
        df["Rank"] = np.nan
        # Assign rank
        # For guides that reside too close to the target's 5' end it may be that not all features are assigned.
        # Thus, all guides with NA features will not be ranked.
        # The rank ranges from 0 to 1, with 1 being the highest rank.
        df.loc[~df["GuideScores"].isna(), "Rank"] = (df.loc[~df["GuideScores"].isna(), "GuideScores"]
                                                     .rank(method="first", ascending=False, na_option="last")
                                                     / df.loc[~df["GuideScores"].isna(), "GuideScores"].count()
                                                     ).round(4).apply(lambda x: 1 - x)
        # Order by rank
        df = df.sort_values("Rank", ascending=False)
    else:
        raise ValueError("Exiting! Guide names do not correspond.")
    # Return object
    return df


def assign_scores(x, q):
    quartiles = []
    for xi in x:
        if np.isnan(xi):
            quartiles.append(np.nan)
        elif xi <= q["25%"]:
            quartiles.append(1)
        elif xi <= q["50%"]:
            quartiles.append(2)
        elif xi <= q["75%"]:
            quartiles.append(3)
        else:
            quartiles.append(4)
    return quartiles


def assign_scores(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Assigns scores to values in x based on the quartile ranges defined by q.
    
    Args:
        x (array): Array of values to assign scores to.
        q (array): Array of quartile ranges.
    
    Returns:
        Array: An array of quartile scores for each value in x. If a value in x is NaN, it will be returned as NaN.
    """
    quartiles = np.digitize(x, q)
    quartiles = [np.nan if np.isnan(xi) else qi for xi, qi in zip(x, quartiles)]
    return quartiles


def get_cds_region_boundaries(x):
    """
    Returns the start and end positions of the coding sequence (CDS) region in a string formatted 
    as "ID:START-END|TYPE", where TYPE can be anything, but is expected to contain "CDS". If no CDS 
    region is found, None is returned.
    
    :param x: A string containing the CDS region.
    :type x: str
    
    :return: A list containing the start and end positions of the CDS region.
    :rtype: list or None
    """
    r = [i for i in x.split("|") if 'CDS' in i]
    if not r:
        return None
    cds = r[0].split(":")[1].split('-')
    return [int(i) for i in cds]


def eval_fold(x: str) -> int:
    """
    Evaluate the input string `x` to determine if it matches the pre-defined 
    RNA fold pattern ((((((.(((....))).)))))). 

    Parameters:
    x (str): The input string to be evaluated.

    Returns:
    int: 1 if the input string matches the pattern, 0 otherwise.
    """
    if x.startswith("((((((.(((....))).))))))"):
        return 1
    else:
        return 0
    
    
def get_mfe(g: str, DR: str, RNAfold_executable: str) -> list:
    """
    Calculates the minimum free energy (MFE) of a given RNA sequence formed by the fusion of the
    direct repeat (DR) and guide RNA (g) using the RNAfold executable.

    :param g: The guide RNA sequence.
    :type g: str
    :param DR: The direct repeat sequence.
    :type DR: str
    :param RNAfold_executable: The path to the RNAfold executable.
    :type RNAfold_executable: str

    :return: A list containing the MFE value, an indicator for whether a G-quadruplex was formed,
             and a dictionary of the DR stem and loop nucleotide pairing.
    :rtype: List[float, int, dict]
    """
    crRNA = DR + g
    cmd = f'echo "{crRNA}" | {RNAfold_executable} --gquad --noPS'
    output = subprocess.check_output(cmd, shell=True, text=True).strip().split("\n")
    fold_output = output[1]
    mfe = float(fold_output.split(" ")[1].strip("()"))
    gq = int("+" in output[1])
    dr = eval_fold(output[1])
    return [mfe, gq, dr]


def read_unpaired_probabilities(x):
    """
    Reads unpaired probabilities from a given file path and returns a pandas DataFrame.
    
    :param x: A file path to a tab-separated file containing unpaired probabilities.
    :type x: str
    
    :return: A transposed pandas DataFrame with unpaired probabilities, with columns 
            labeled numerically from 1 to 50 and indexed by row number.
    :rtype: pandas.DataFrame
    """
    up_cols = ["X.i."] + list(range(1, 51))
    UnpairedProbabilities = pd.read_csv(x, sep="\t", skiprows=1, index_col="X.i.", usecols=up_cols)
    UnpairedProbabilities.columns = range(1, 51)
    UnpairedProbabilities = UnpairedProbabilities.T
    return UnpairedProbabilities


def transform_RNAplfold_predictions(x: np.ndarray) -> np.ndarray:
    """
    Transforms RNAplfold predictions by filling a matrix where the first column is taken from the first column of x, and the rest of the columns are shifted one row down and to the right. 
    
    Args:
    - x (pandas.DataFrame): A DataFrame representing RNAplfold predictions.
    
    Returns:
    - matrix (numpy.ndarray): A 2D array where the first column is the first column of x, and the rest of the columns are shifted one row down and to the right. Any missing values are filled with NaN.
    """
    matrix = np.full_like(x, np.nan)
    matrix[:, 0] = x[:, 0]
    for i in range(1, x.shape[0]):
        d = (i - 1) // 2 if i % 2 != 0 else i // 2
        matrix[i, 1 + d:] = x[i, :-1 - d]
    return matrix


def slice_matrix(center_idx: int, matrix: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
    """
    Slice a matrix based on a given center index and a window size.

    :param center_idx: The index around which to slice the matrix.
    :type center_idx: int or float

    :param matrix: The matrix to be sliced.
    :type matrix: pandas DataFrame

    :param window_size: The size of the window around the center index to slice.
    :type window_size: int

    :return: A sliced version of the matrix, with the same index as the original matrix and columns ranging from
        -window_size to window_size.
    :rtype: pandas DataFrame
    """
    if np.isnan(center_idx):
        return pd.DataFrame(np.nan, index=matrix.index, columns=range(-window_size, window_size+1))
    left_idx = max(center_idx-window_size, 0)
    right_idx = min(center_idx+window_size+1, matrix.shape[1])
    sliced_matrix = matrix.iloc[:, left_idx:right_idx]
    if center_idx-window_size < 1:
        slice_idx = slice(None, center_idx+window_size)
        sliced_matrix.iloc[:, window_size-center_idx+1:] = matrix.iloc[:, slice_idx].loc[:, :sliced_matrix.columns[0]-1]
    elif center_idx+window_size >= matrix.shape[1]:
        slice_idx = slice(center_idx-window_size, None)
        sliced_matrix.iloc[:, :window_size+1+(matrix.shape[1]-center_idx)] = matrix.iloc[:, slice_idx].loc[:, sliced_matrix.columns[-1]+1:]
    return sliced_matrix


def get_value(x: pd.DataFrame, d: int, w: int) -> float:
    """
    Return the value at index (d, w) of a pandas DataFrame x, or np.nan if x is empty.

    Parameters:
    x (pandas.DataFrame): The DataFrame to extract the value from.
    d (int): The row index of the desired value.
    w (int): The column index of the desired value.

    Returns:
    The value at index (d, w) of x, or np.nan if x is empty.
    """
    if x.shape[0] == 0:
        return np.nan
    else:
        return x.iloc[d, w]
    
    
def get_unpaired_probabilities(
    x: pd.DataFrame, 
    matrix: pd.DataFrame, 
    window_size: int = 50, 
    density_window: int = 40, 
    density_distance: int = 23) -> pd.Series:
    """
    Calculates the unpaired probabilities of nucleotide density for each guide in the input x.
    
    :param x: A dictionary containing match positions of nucleotides.
    :type x: dict
    
    :param matrix: A matrix containing nucleotide density.
    :type matrix: numpy.ndarray
    
    :param window_size: The size of the window to slice the nucleotide density matrix (default is 50).
    :type window_size: int
    
    :param density_window: The size of the window to calculate the value of density (default is 40).
    :type density_window: int
    
    :param density_distance: The distance between the two windows to calculate the density value (default is 23).
    :type density_distance: int
    
    :return: A pandas series containing the unpaired probabilities of nucleotide density for each guide.
    :rtype: pandas.Series
    """
    # slice the nucleotide density matrix for each guide to obtain a window of +/- the window size (default 50nt) centered on guide match position 1
    slices = {k: slice_matrix(v, matrix=matrix, window_size=window_size) for k, v in x["MatchPos"].items()}
    densities = pd.Series({k: get_value(v, density_distance, density_window) for k, v in slices.items()})
    return densities


def get_unpaired_probabilities_sample(x: pd.DataFrame, matrix: pd.DataFrame) -> pd.Series:
    """
    Returns a Pandas Series containing the densities of unpaired probabilities.
    
    Args:
    - x: A Pandas DataFrame representing a set of data.
    - matrix: A Pandas DataFrame representing a matrix of data.
    
    Returns:
    - A Pandas Series representing the densities of unpaired probabilities.
    """
    # Get relevant row from Matrix
    density_vec = matrix.iloc[23, :]
    # Get value relative to guide match position (offset of -11 relative to start position)
    match_pos = x["MatchPos"] - 11
    densities = density_vec.loc[match_pos].reset_index(drop=True)
    return densities


def get_target_site_accessibility(data: pd.DataFrame, fasta: SeqIO, RNAplfold: str, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Returns a Pandas Series containing the densities of unpaired probabilities.
    
    Args:
    - x: A Pandas DataFrame representing a set of data.
    - matrix: A Pandas DataFrame representing a matrix of data.
    
    Returns:
    - A Pandas Series representing the densities of unpaired probabilities.
    """
    # Generate random string for tmp file that will be written to the hard drive to avoid collisions
    logger.info(f'Calculating target site accessibility ...')
    random_string = "".join(random.choices(string.ascii_lowercase, k=6))
    fasta.id = random_string
    # Writing the fasta file back to hard drive as a tmp file. This is done to be independent of any naming issues
    SeqIO.write(fasta, f"./{random_string}.fa", "fasta")
    # You may need to change the path to your RNAplfold executable
    cmd = f"cat {random_string}.fa | {RNAplfold} -L 40 -W 80 -u 50"
    output = subprocess.check_output(cmd, shell=True, text=True)
    unpaired_probabilities = read_unpaired_probabilities(f"./{random_string}_lunp")
    unpaired_probabilities_transformed = transform_RNAplfold_predictions(unpaired_probabilities)
    # As there was no clear pattern, this will only record the unpaired probability covering the entire guide match
    log10_unpaired = get_unpaired_probabilities_sample(data, MA=log10(unpaired_probabilities_transformed))
    # Clean up
    os.remove(f"./{random_string}.fa")
    os.remove(f"./{random_string}_lunp")
    os.remove(f"./{random_string}_dp.ps")
    return log10_unpaired


def get_RNAhyb_mfe(sequence: str) -> float:
    """
    Compute the minimum free energy (MFE) of the hybrid formed by RNAhybrid
    between a given target sequence and a candidate sequence. 

    :param sequence: A string representing the candidate sequence.
    :return: A float representing the MFE of the RNAhybrid formed between
        the target sequence and the candidate sequence.
    """
    cmd = f"RNAhybrid -c -s 3utr_human {sequence} {reverse_complement(sequence)}"
    output = subprocess.check_output(cmd, shell=True).decode()
    mfe = float(output.splitlines()[2].split()[4][1:-1])
    return mfe


def get_RNAhyb_mfe_bulk(
    data: pd.DataFrame, 
    RNAhyb: str, 
    position: int = 3, 
    width: int = 12, 
    logger: logging.Logger = None) -> pd.DataFrame:
    """
    Calculates the minimum free energy (MFE) of RNAhyb in bulk for a given set of guide and target sequences.
    
    :param data: A pandas DataFrame containing the guide sequences as a column named "GuideSeq".
    :type data: pd.DataFrame
    :param RNAhyb: The path to the RNAhyb executable used to calculate MFE.
    :type RNAhyb: str
    :param position: The index position of the guide sequence to start the MFE calculation.
    :type position: int, optional
    :param width: The width of the guide sequence to be used in the MFE calculation.
    :type width: int, optional
    :param logger: A logging.Logger object to record progress and errors.
    :type logger: logging.Logger, optional
    :return: A pandas DataFrame containing the calculated MFE values for each guide sequence.
    :rtype: pd.DataFrame
    """
    # Transform to DNAStringSet
    logger.info(f'Calculating RNAhyb MFE ...')
    dna_seqs = DNAStringSet(data["GuideSeq"])
    # Extract guide and target
    guide_seq = str(dna_seqs[:, position:(position+width-1)])
    target_seq = str(reverse_complement(dna_seqs[:, position:(position+width-1)]))
    # Write tmp file to hard disk
    tmp_df = pd.DataFrame({"guide": guide_seq, "target": target_seq})
    random_string = "".join(random.choices(string.ascii_lowercase, k=6))
    tmp_df.to_csv(f"{random_string}_hyb_mfe_{position}.{width}.txt", sep=",", index=False, header=False)
    # Calculate RNA hyb MFE in bulk
    cmd = f"bash {RNAhyb} {random_string}_hyb_mfe_{position}.{width}.txt"
    output = subprocess.check_output(cmd, shell=True, text=True)
    # Extract the MFE
    hyb_mfe = [float(x.split()[4]) for x in output.splitlines()]  # Gets MFE
    # Clean up
    os.remove(f"{random_string}_hyb_mfe_{position}.{width}.txt")
    # Return value
    return hyb_mfe


def get_letter_probs(x: pd.DataFrame, s: int = 1, e: int = 23):
    """
    This function calculates the probabilities of each letter, di-nucleotide, and nucleotide pairs in the given guide sequences.
    :param x: A pandas DataFrame containing the guide sequences.
    :param s: An integer representing the start position of the guide sequence.
    :param e: An integer representing the end position of the guide sequence.
    :return: A pandas DataFrame with probabilities of each letter, di-nucleotide, and nucleotide pairs in the guide sequences.
    """
    g = x['GuideSeq'].str.slice(s-1, e-1)

    a_prob = g.str.count('A') / g.str.len()
    c_prob = g.str.count('C') / g.str.len()
    g_prob = g.str.count('G') / g.str.len()
    t_prob = g.str.count('T') / g.str.len()
    gc_prob = g.str.count('GC') / (g.str.len()-1)
    au_prob = g.str.count('AT') / (g.str.len()-1)

    di_nucleotide_prob = pd.concat([g.str[i:i+2].value_counts(normalize=True) for i in range(g.str.len()-1)], axis=1).fillna(0).T

    out = pd.concat([a_prob, c_prob, g_prob, t_prob, gc_prob, au_prob, di_nucleotide_prob], axis=1)

    out.columns = ['pA', 'pC', 'pG', 'pT', 'pG|pC', 'pA|pT', 'pAA', 'pAC', 'pAG', 'pAT',
                   'pCA', 'pCC', 'pCG', 'pCT', 'pGA', 'pGC', 'pGG', 'pGT', 'pTA', 'pTC', 'pTG', 'pTT']
    out.index = x.index

    return out


def gg_plot(df: pd.DataFrame, length: int, name: str=None, cds: str=None, factor: float=0.2, filename: str=None) -> None:
    """
    This function plots a scatter plot with the guide match position and the standardized guide score of different quartiles.
    The function takes a pandas dataframe as its first argument which contains the data to be plotted. The second argument 
    is the length of the plot, and is an integer. The third argument is an optional string which is the title of the plot. 
    The fourth argument is also an optional string of coordinates to overlay on the plot, which is formatted as a string of 
    comma-separated pairs of integers. The fifth argument is an optional float which is the scaling factor of the plot length. 
    The function returns nothing, but displays the plot on the screen.
    :param df: A pandas DataFrame containing data to be plotted.
    :param length: An integer representing the length of the plot.
    :param name: An optional string representing the name of the plot.
    :param cds: An optional string of coordinates to overlay on the plot, formatted as a string of comma-separated pairs of integers.
    :param factor: An optional float representing the scaling factor of the plot length.
    :return: None
    """
    df = df.dropna(subset=["quartiles"])
    if length < 500:
        bin_size = 25
    elif 500 <= length < 1000:
        bin_size = 50
    elif 1000 <= length < 2000:
        bin_size = 100
    elif 2000 <= length < 5000:
        bin_size = 500
    elif 5000 <= length < 10000:
        bin_size = 1000
    else:
        bin_size = 4000
    if name is None:
        name = ""
    if cds is None:
        max_score = df["standardizedGuideScores"].max()
        min_score = df["standardizedGuideScores"].min()
        score_range = max_score - min_score
        df["quartiles"] = pd.Categorical(df["quartiles"], categories=df["quartiles"].unique())
        ymax = 1
        ymin = 0
        fig, ax = plt.subplots()
        for quartile, data in df.groupby("quartiles"):
            ax.scatter(data["MatchPos"], data["standardizedGuideScores"], label=quartile, s=20)
        x = np.linspace(0, length, 100)
        y = np.zeros_like(x)
        ax.plot(x, y, color="#525252")
        ax.set_title(name)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("guide match position [nt]")
        ax.set_ylabel("standardized guide score")
        ax.set_xlim(0, length)
        ax.set_xticks(range(0, length+1, bin_size))
        ax.legend(title="Quartiles", loc="upper right")
        fig.set_size_inches(length*factor/100, 8)
    else:
        cds_df = pd.DataFrame(cds, columns=["start", "end"])
        max_score = df["standardizedGuideScores"].max()
        min_score = df["standardizedGuideScores"].min()
        score_range = max_score - min_score
        df["quartiles"] = pd.Categorical(df["quartiles"], categories=df["quartiles"].unique())
        ymax = 1
        ymin = 0
        fig, ax = plt.subplots()
        for quartile, data in df.groupby("quartiles"):
            ax.scatter(data["MatchPos"], data["standardizedGuideScores"], label=quartile, s=20)
        x = np.linspace(0, length, 100)
        y = np.zeros_like(x)
        ax.plot(x, y, color="#525252")
        ax.set_title(name)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("guide match position [nt]")
        ax.set_ylabel("standardized guide score")
        ax.set_xlim(0, length)
        ax.set_xticks(range(0, length+1, bin_size))
        ax.legend(title="Quartiles", loc="upper right")
        fig.set_size_inches(length*factor/100, 8)
        for i in range(len(cds_df)):
            ax.axvspan(cds_df["start"][i], cds_df['end'][i], alpha=0.2, color='#969696')
            ax.text(cds_df['start'][i], 0.01, 'CDS', ha='left', va='bottom', fontsize=12)
    plt.savefig(filename)
    plt.show()
    
    
    
def read_and_check_fasta(inpath: str, logger: logging.Logger) -> list:
    """
    Reads a FASTA file and checks the length of each sequence. Sequences with length >= 30 are added to the 
    `sequences` list and sequences with length < 80 are added to the `short_seqs` list. If `short_seqs` is not 
    empty, a message is printed indicating that the minimum length for the target site accessibility calculation 
    is 80nt. The list of short sequences is also printed along with their IDs and lengths. If `sequences` is empty, 
    the program is terminated with a message indicating that a long sequence is required to assess the target site 
    nucleotide context. Returns the list of sequences with length >= 30.
    
    :param inpath: A string representing the input file path.
    :type inpath: str
    :return: A list of sequences (with length >= 30) that were read from the input file.
    :rtype: list
    """
    fasta = SeqIO.read(inpath, "fasta")
    # 遍历每个序列,检查每个序列的长度,提示长度小于80和30的
    sequences = []
    short_seqs = []
    for fa in fasta:
        if len(fa.seq) >= 30:
            sequences.append(fa)
        if len(fa.seq) < 80:
            short_seqs.append(fa)
    if len(short_seqs) != 0:
        logger.info("RNAplfold requires a minimum of 80nt length for the target site accessibility calculation")
        logger.info("Please provide an input sequence >30nt to be able to assess the target site nucleotide context.")
        logger.info("Short sequences:")
        for fa in short_seqs:
            logger.info(f'\t{fa.id}: {len(fa.seq)}')
        if len(sequences) == 0:
            logger.error('Do not got any sequence to be able to assess the target site nucleotide context.')
            sys.exit('Exiting! Please provide an long sequence to be able to assess the target site nucleotide context.')
    return sequences


def get_guides_for_sequence(
    sequence: Seq.SeqRecord, 
    guide_length: int=25, 
    gc_min: float=0.2, 
    gc_max: float=0.8, 
    homopolymer_length_t: int=3, 
    homopolymer_length_nont: int=4,
    logger: logging.Logger=None) -> list:
    """
    Get possible guides for a given DNA sequence.

    Args:
        sequence (Seq.SeqRecord): A DNA sequence to get guides for.
        guide_length (int, optional): Length of guides. Defaults to 25.
        gc_min (float, optional): Minimum allowed GC content for guides. 
            Defaults to 0.4.
        gc_max (float, optional): Maximum allowed GC content for guides. 
            Defaults to 0.6.
        homopolymer_length_t (int, optional): Maximum allowed homopolymer 
            length for thymine. Defaults to 3.
        homopolymer_length_nont (int, optional): Maximum allowed homopolymer 
            length for non-thymine bases. Defaults to 4.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        list: A list of possible guides for the given DNA sequence.
    """
    logger.info(f"Getting guides for sequence: {sequence.id}")
    logger.info(f"Get raw CasRx guides sequences started on {datetime.now()}")
    raw_guides = get_all_possible_guides(
        seq = str(sequence.seq), 
        guide_length = guide_length, 
        gc_min = gc_min, 
        gc_max = gc_max, 
        homopolymer_length_t = homopolymer_length_t, 
        homopolymer_length_nont = homopolymer_length_nont
        )
    return raw_guides


# RUN ---------------------------------------------------------------------------
if __name__ == '__main__':
    logger = set_logger('CasRxDesigner', level = 'DEBUG')
    options= get_options(logger=logger)
    # 读取选定的序列
    cdna_seqs = read_and_check_fasta(options.fasta, logger=logger)
    # 获取所有潜在的gRNA
    cdna_guides_fwd = {}
    cdna_guides_rev = {}
    for cdna in cdna_seqs:
        cdna_guides_fwd[cdna.id] = get_guides_for_sequence(
            sequence=cdna, 
            guide_length=options.guide_length, 
            gc_min=options.gc_min, 
            gc_max=options.gc_max, 
            homopolymer_length_t=options.homopolymer_length_t, 
            homopolymer_length_nont=options.homopolymer_length_nont, 
            logger=logger
            )
        cdna_guides_rev[cdna.id] = [reverse_complement(seq) for seq in cdna_guides_fwd[cdna.id]]
        
    
        
    