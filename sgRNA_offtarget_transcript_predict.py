# /root/miniconda3/envs/wes/bin/python
# created  : 2022-8-16
# last_edit: 2023-3-22
# author   : liozh

from __future__ import division
from optparse import OptionParser
import datetime
import re
import os
from numpy import isin
from tqdm import tqdm
from Bio import SeqIO
from Bio import AlignIO
from Bio import Align
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Emboss.Applications import WaterCommandline
from Bio import pairwise2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool
from collections import Counter
from collections import defaultdict
import pandas as pd
import subprocess
import gzip


# function ----------
def getOptions():
    usage = (
            """
            python this_scripy.py -m ALL -@ 5 -n 21 -u 6 \\
                                        --gapopen 10.0 \\
                                        --gapextend 0.5 \\
                                        -r cdnd.fa \\
                                        -o out_dir/prefix \\
                                        -s sgrna.fa \\
                                        -t target_cdna_id \\
                                        -a both

            执行步骤：
            1）将sgRNA FASTA文件内的序列按每个文件一条序列进行拆分，每条序列分别获取反向互补序列。
            2）通过EMBOSS::WATER程序将每条单独的sgRNA序列与给定的转录组cDNA序列比对。
            3）通过给定的参数过滤比对结果，获取match及unmatch在一定范围内的比对序列。
            
            说明：
            --gapopen 及 --gapextend 罚分设置请参考 https://www.bioinformatics.nl/cgi-bin/emboss/help/water
            """
    )
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-m",
        "--mode",
        dest="mode",
        type="string",
        metavar="STR",
        default=False,
        help="Running mode.")
    parser.add_option(
        "-r",
        "--cdna",
        dest="cdna_path",
        type="string",
        metavar="FILE",
        default=False,
        help="输入FASTA格式的目标转录组文件.")
    parser.add_option(
        "-o",
        "--output",
        dest="prefix",
        type="string",
        metavar="STR",
        default="./",
        help="输出文件的路径及前缀.")
    parser.add_option(
        "-p",
        "--gapopen",
        dest="gapopen",
        type="float",
        metavar="FLOAT",
        default=10.0,
        help="gap罚分.")
    parser.add_option(
        "-e",
        "--gapextend",
        dest="gapextend",
        type="float",
        metavar="FLOAT",
        default=0.5,
        help="extend罚分.")
    parser.add_option(
        "-@",
        "--threads",
        dest="threads",
        type="int",
        metavar="INT",
        default=2,
        help="运行所使用的线程数.")
    parser.add_option(
        "-s",
        "--sgrna",
        dest="sgrna_path",
        type="string",
        metavar="FILE",
        default=False,
        help="FASTA格式的sgRNA文件.")
    parser.add_option(
        "-n",
        "--min_match",
        dest="min_match",
        type="int",
        metavar="INT",
        default=21,
        help="sgRNA与转录组的最小比对碱基数.")
    parser.add_option(
        "-u",
        "--max_unmatch",
        dest="max_unmatch",
        type="int",
        metavar="INT",
        default=6,
        help="sgRNA与转录组的最大不匹配碱基数.")
    parser.add_option(
        "-t",
        "--target",
        dest="target",
        type="string",
        metavar="ENSEMBL_TRANSID",
        default=False,
        help="转录组cDNA文件中的靶标序列ID或名称，比如 {ENST00000620773.1|VEGFA} .")
    parser.add_option(
        "-a",
        "--strand",
        dest="strand",
        type="string",
        metavar="strand",
        default='both',
        help="sgRNA使用正向或反向互补链进行比对. {both|fwd|rev}"),
    parser.add_option(
        "-c",
        "--checkPackages",
        dest="checkPackages",
        type="string",
        metavar="bool",
        default=False,
        help="检查软件依赖是否正确安装. {True|False}")

    (options, args) = parser.parse_args()
    if options.checkPackages == 'True':
        checkPackages()
        exit()
    if not (options.cdna_path or options.sgrna_path):
        parser.print_help()
        exit()
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
        os.rmdir(options.prefix)
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time} <PARAM>  {options}')
    return options


def checkPackages():
    """
    Check if required packages are installed, if not, install them automatically
    """
    packages = ['emboss', 'pandas', 'biopython', 'tqdm']
    subprocess.check_call(["mamba", "install", "-c", "bioconda", "-c", "conda-forge"] + packages)
    print("All required packages are installed.")


def grepTargetCdna(opt):
    """
    获取靶标的cDNA序列并输出到文件，返回文件的地址
    """
    target_list = opt.target.split(",") if opt.target != "False" else []
    cdna_path = f'{opt.prefix}_target.cdna.fa'
    cdna_seq = SeqIO.parse(opt.cdna_path, "fasta")
    with open(cdna_path, "w") as cdna_handle:
        for record in cdna_seq:
            if record.id in target_list or any(target in record.description for target in target_list):
                SeqIO.write(record, cdna_handle, "fasta")
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time} <GREP>   get target cDNA sequence:{cdna_path}')
    return cdna_path


def readSequence(sequence_path, to_str=False):
    seqs = []
    with open(sequence_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            record = SeqRecord(record.seq.upper(), id=record.id, description=record.description)
            seqs.append(record)
    if to_str:
        return [str(seq.seq) for seq in seqs]
    else:
        return seqs


def splitSgrna(sgrna_path, out_path, strand="both"):
    """
    将sgRNA的FASTA序列切分为一条序列一个文件，并取反向互补序列
    """
    sglist = []
    sgrna_seq = SeqIO.parse(sgrna_path, 'fasta')
    for record in sgrna_seq:
        record.seq = record.seq.upper()
        sgrna_name = record.id.split(":")[0]
        swater = f'{out_path}_{sgrna_name}.water'

        fwd_fa = f'{out_path}_water_sg_fwd_{sgrna_name}.fasta'
        fwd_water = f'{out_path}_water_sg_fwd_{sgrna_name}.water'
        fmesg = [sgrna_name, record.seq, fwd_fa, fwd_water, swater]

        rev_fa = f'{out_path}_water_sg_rev_{sgrna_name}.fasta'
        rev_water = f'{out_path}_water_sg_rev_{sgrna_name}.water'
        rmesg = [sgrna_name, record.seq.reverse_complement(), rev_fa, rev_water, swater]

        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if strand == "fwd":
            sglist.append(fmesg)
            record.seq = record.seq
            SeqIO.write(record, fwd_fa, "fasta")
            print(f'{time} <PARAM>  {fwd_fa} {fmesg[1]}')
        elif strand == "rev":
            sglist.append(rmesg)
            record.seq = record.seq.reverse_complement()
            SeqIO.write(record, rev_fa, "fasta")
            print(f'{time} <PARAM>  {rev_fa} {rmesg[2]}')
        elif strand == "both":
            sglist.append(fmesg)
            SeqIO.write(record, fwd_fa, "fasta")
            print(f'{time} <PARAM>  {fwd_fa} {fmesg[1]}')
            sglist.append(rmesg)
            record.seq = record.seq.reverse_complement()
            SeqIO.write(record, rev_fa, "fasta")
            print(f'{time} <PARAM>  {rev_fa} {rmesg[2]}')
    return sglist


def getWaterAlignment(sgrna_path, cdna_path, out_path, gapopen=10, gapextend=0.5):
    """
    将切分的sgRNA序列与转录本比对
    """
    wcline = WaterCommandline(
        gapopen=gapopen,
        gapextend=gapextend,
        asequence=sgrna_path,
        bsequence=cdna_path,
        outfile=out_path
    )
    # print(wcline)
    stdout, stderr = wcline()
    # print(stdout + stderr)
    return out_path


def poolGetWaterAlignment(sgrna_list, cdna_path, gapopen=10, gapextend=0.5, threads=2, realign=False):
    pool = Pool(processes=threads)
    if not realign:
        params = [(record[2], cdna_path, record[3], gapopen, gapextend) for record in sgrna_list if not os.path.exists(record[3])]
    else:
        params = [(record[2], cdna_path, record[3], gapopen, gapextend) for record in sgrna_list]
    results = []
    with tqdm(desc='Aligning', total=len(params)) as pbar:
        for param in params:
            result = pool.apply_async(func=getWaterAlignment, args=param, callback=lambda x: pbar.update())
            # results.append(result.get())
            results.append(result)
        results = [result.get() for result in results]
        pool.close()
        pool.join()
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time} <OUTPUT> {results}')
    return results


def getCigarString(alignments, start, end):
    cigar = ''
    seq1 = alignments[0]
    seq2 = alignments[1]
    for i in range(start, end):
        if seq1[i] == seq2[i]:
            cigar += '|'
        elif seq1[i] == '-':
            cigar += 'I'
        elif seq2[i] == '-':
            cigar += 'D'
        else:
            cigar += 'X'
    return cigar


# def getOffTargets(cdna_seq, sgRNA_seq, max_mismatches, min_consecutive_matches, num_processes=None,num_threads=None):
#     off_targets = []
#     if num_processes:
#         with Pool(num_processes) as pool:
#             for sgRNA, cdna in tqdm(itertools.product(sgRNA_seq, cdna_seq), total=len(sgRNA_seq)*len(cdna_seq), desc='sgRNA'):
#                 forward_result = pool.apply_async(pairwiseLocalAlign, args=(cdna, sgRNA, max_mismatches, min_consecutive_matches))
#                 reverse_result = pool.apply_async(pairwiseLocalAlign, args=(cdna, reverseComplement(sgRNA), max_mismatches, min_consecutive_matches))
#                 forward_result = forward_result.get()
#                 reverse_result = reverse_result.get()
#                 if forward_result:
#                     off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '+', **forward_result})
#                 if reverse_result:
#                     off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '-', **reverse_result})
#         return pd.DataFrame(off_targets)
#     elif num_threads:
#         with ThreadPoolExecutor(max_workers=num_threads) as executor:
#             for sgRNA, cdna in tqdm(itertools.product(sgRNA_seq, cdna_seq), total=len(sgRNA_seq)*len(cdna_seq), desc='sgRNA'):
#                 forward_result = executor.submit(pairwiseLocalAlign, cdna, sgRNA, max_mismatches, min_consecutive_matches)
#                 reverse_result = executor.submit(pairwiseLocalAlign, cdna, reverseComplement(sgRNA), max_mismatches, min_consecutive_matches)
#                 forward_result = forward_result.result()
#                 reverse_result = reverse_result.result()
#                 if forward_result:
#                     off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '+', **forward_result})
#                 if reverse_result:
#                     off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '-', **reverse_result})
#         return pd.DataFrame(off_targets)
#     else:
#         for sgRNA, cdna in tqdm(itertools.product(sgRNA_seq, cdna_seq), total=len(sgRNA_seq)*len(cdna_seq), desc='sgRNA'):
#             forward_result = pairwiseLocalAlign(cdna, sgRNA, max_mismatches, min_consecutive_matches)
#             reverse_result = pairwiseLocalAlign(cdna, reverseComplement(sgRNA), max_mismatches, min_consecutive_matches)
#             if forward_result:
#                 off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '+', **forward_result})
#             if reverse_result:
#                 off_targets.append({'sgRNA': sgRNA.id, 'cdna': cdna.id, 'strand': '-', **reverse_result})
#         return pd.DataFrame(off_targets)
    
    
def pairwiseLocalAlign(seq1, seq2, max_mismatches=None, min_consecutive_matches=None):
    try:
        alignments = pairwise2.align.localms(seq1.seq, seq2.seq, 2, -1, -8, -0.5, one_alignment_only=True, gap_char='-', force_generic=True)
        cigar = getCigarString(alignments[0], alignments[0].start, alignments[0].end)
        matches = cigar.count('|')
        mismatches = cigar.count('X') + cigar.count('I') + cigar.count('D')
        start = alignments[0].start
        end = alignments[0].end
        max_consecutive_matches = max([len(x) for x in re.split('[DXI]', cigar)])
        if (max_mismatches and mismatches > max_mismatches) or (min_consecutive_matches and max_consecutive_matches < min_consecutive_matches):
            return None
        else:
            return {'matches': matches, 'mismatches': mismatches, 'start': start, 'end': end,
                    'max_consecutive_matches': max_consecutive_matches, 
                    'seq1':alignments[0][0][start:end], 'seq2':alignments[0][1][start:end], 'cigar':cigar}
    except Exception as e:
        print(e)
        return None
    
def reverseComplement(seq):
    if isinstance(seq, str):
        seq = seq.translate(str.maketrans("ATCG", "TAGC"))[::-1]
    else:
        seq.seq = seq.seq.reverse_complement()
    return seq


def calMismatchAndGap(cdna, sgrna, sgrna_len):
    cdna_seq = cdna.seq
    sgrna_seq = sgrna.seq
    unalign = 0
    match = 0
    max_seed = []
    seed = 0
    for cdna_base, sgrna_base in zip(cdna_seq, sgrna_seq):
        if cdna_base != sgrna_base:
            unalign += 1
            max_seed.append(seed)
            seed = 0
        else:
            seed += 1
            match += 1
    max_seed.append(seed)
    unmatch = sgrna_len - match
    max_seed = max(max_seed)
    return match, unmatch, max_seed


def filterWaterResult(water_out_path, sgrna_len, min_match, max_unalign, sgrna_id):
    water = list(AlignIO.parse(water_out_path, 'emboss'))
    water_out = []
    with tqdm(total=len(water), desc=f'Filtering {sgrna_id}') as pbar:
        for record in water:
            sgrna, cdna = record
            match, unmatch, max_seed_len = calMismatchAndGap(cdna, sgrna, sgrna_len)
            align_len = record.get_alignment_length()
            if max_seed_len >= 8 and match >= sgrna_len - max_unalign and match >= min_match and sgrna_len + max_unalign >= align_len:
                mesg = [sgrna_id, cdna.id, sgrna.seq, cdna.seq, sgrna_len, match, unmatch, max_seed_len, align_len]
                water_out.append(mesg)
            pbar.update(1)
    # time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # print(f'{time} <RESULT> {sgrna_id} got {len(water_out)} results')
    return water_out


def poolFilterWaterResult(sgrna_list, min_match, max_unalign, threads=2):
    pool = Pool(processes=threads)
    results = []
    for record in tqdm(sgrna_list, desc='Filtering result'):
        water_path = record[3]
        sgrna_len = len(record[1])
        sgrna_id = record[0]
        result = pool.apply_async(filterWaterResult, args=(water_path, sgrna_len, min_match, max_unalign, sgrna_id))
        results.append(result)
    results = [result.get() for result in results]
    pool.close()
    pool.join()
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time} <RESULT> got {len(results)} different sgRNA result')
    return results


def removeTmpfile(sgrna_list):
    gzfile = []
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{time} <RESULT> Creating gzipfile for result and remove tmpfile.')
    with tqdm(total=len(sgrna_list), desc='Removing tmpfile') as pbar:
        for record in sgrna_list:
            sgrna_fa = record[2]
            water_out = record[3]
            gzipout = f'{water_out}.gz'
            if os.path.exists(gzipout):
                gzfile.append(gzipout)
                pbar.update(1) 
                continue
            CMD = f'gzip {water_out} -c > {gzipout}'
            subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            try:
                os.remove(sgrna_fa)
                os.remove(water_out)
                gzfile.append(gzipout)
            except:
                continue
            pbar.update(1)        
    return gzfile


def tarFiles(file_list, output_file):
    try:
        subprocess.run(['tar', '-czvf', output_file] + file_list)
        for file in file_list:
            os.remove(file)
    except:
        pass


def mergeToDataframe(multi_list):
    alist = []
    for res in multi_list:
        alist.extend(res)
    df = pd.DataFrame(
        alist,
        columns=['sgRNA_ID', 'Transcript_ID', 'sgRNA_align', 
                 'Transcript_align', 'sgRNA_len', 'Match', 'Unmatch',
                 'Max_seed', 'Alignment_length']
    )
    df.drop_duplicates()
    df.sort_values(
        by=["sgRNA_ID", "Unmatch", "Transcript_ID"], 
        ascending=[False, True, False], 
        inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def getIDs(line):
    gene = None
    trans = line.split(" ")[0]
    syml = None
    chrom = None
    trans_type = None
    gene_p = r'gene:(.*?)$'
    syml_p = r'gene_symbol:(.*?)$'
    chrom_p = r'chromosome:(.*?)$'
    trans_p = r'transcript_biotype:(.*?)$'
    for elem in line.split(" "):
        gene_m = re.search(gene_p, elem)
        syml_m = re.search(syml_p, elem)
        chrom_m = re.search(chrom_p, elem)
        trans_m = re.search(trans_p, elem)
        if gene_m:
            gene = gene_m.group(1)
        elif syml_m:
            syml = syml_m.group(1)
        elif chrom_m:
            chrom = chrom_m.group(1)
            c = chrom.split(":")
            chrom = f'{c[1]}:{c[2]}-{c[3]}:{c[4]}'
        elif trans_m:
            trans_type = trans_m.group(1)
    return [trans, gene, syml, chrom, trans_type]


def getTransformat(cdna_seq):
    anno = []
    for record in cdna_seq:
        des = record.description
        fmt = getIDs(des)
        fmt.append(len(str(record.seq)))
        anno.append(fmt)
    df = pd.DataFrame(anno, columns=['Transcript_ID', 'Gene_ID', 'Gene_Symbol', 'Gene_Position', 'Transcript_Biotype', 'Transcript_Length'])
    df.drop_duplicates(inplace=True)
    df.sort_values(by=["Gene_Symbol", "Transcript_Length", "Transcript_ID", "Gene_Position"], ascending=[True, False, True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def annoTranscript(data=None, data_path=None, trans_anno=None, cdna_path=None):
    if data_path and not data:
        data =  pd.read_csv(data_path, sep="\t")
    if cdna_path and not trans_anno:
        cdna_fa = SeqIO.parse(cdna_path, "fasta")
        trans_anno = getTransformat(cdna_fa)
    data = pd.merge(data, trans_anno, how='inner', on='Transcript_ID')
    data.drop_duplicates(inplace=True)
    data.sort_values(by=["sgRNA_ID", 'Max_seed',"Match"], ascending=[False, False, False], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data



def main():
    options = getOptions()
    if options.target:
        print(options.target)
        target_cdna_path = grepTargetCdna(options)
        cdna_seq = readSequence(target_cdna_path)
        cdna_anno = getTransformat(cdna_seq)
    else:
        target_cdna_path = options.cdna_path
        cdna_seq = readSequence(target_cdna_path)
        cdna_anno = getTransformat(cdna_seq)

    sgRNA_list = splitSgrna(
        sgrna_path=options.sgrna_path,
        out_path=options.prefix,
        strand=options.strand
    )

    poolGetWaterAlignment(
        sgrna_list=sgRNA_list,
        cdna_path=target_cdna_path,
        gapopen=options.gapopen,
        gapextend=options.gapextend,
        threads=options.threads,
        realign=False
    )

    flt_list = poolFilterWaterResult(
        sgrna_list=sgRNA_list,
        min_match=options.min_match,
        max_unalign=options.max_unmatch,
        threads=options.threads
    )

    flt_df = mergeToDataframe(flt_list)
    flt_file = f'{options.prefix}_water_fltM{options.min_match}U{options.max_unmatch}.result.txt'
    flt_df.to_csv(flt_file, sep="\t", index=False, encoding='utf-8')

    # annotated transcript
    anno_df = annoTranscript(
        data=flt_df, 
        data_path=None, 
        trans_anno=cdna_anno, 
        cdna_path=None
        )
    anno_file = f'{options.prefix}_water_fltM{options.min_match}U{options.max_unmatch}.result.annotated.txt'
    anno_df.to_csv(anno_file, sep="\t", index=False, encoding='utf-8')

    tar_file = f'{options.prefix}_water_fltM{options.min_match}U{options.max_unmatch}.result.tar.gz'
    gz_files = removeTmpfile(sgrna_list=sgRNA_list)
    tarFiles(gz_files, tar_file)
    print(f'Output: {tar_file}')

    times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{times} <RESULT> Done !!!')


# run ----------
# sgRNA_seq_path = '/mnt/d/desktop/RNAseq_zh04_20220801_human/sgRNA_offtarget_predict/mismatch4/sgRNA_seq.fa'
# trans_seq_path = '/mnt/d/download/human_genome/hg38/Homo_sapiens.GRCh38.cdna.all.fa'
# sgRNA_seq = SeqIO.parse(sgRNA_seq_path, 'fasta')
# transcript_seq = SeqIO.parse(trans_seq_path, 'fasta')
# sgRNA_list = list(sgRNA_seq)

if __name__ == '__main__':
    main()
