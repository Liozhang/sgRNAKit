from re import L
import requests
import sys
import argparse
import pandas as pd

# 定义函数加载输入参数
def get_options():
    parser = argparse.ArgumentParser(
        description="locate cdna to dna",
        epilog="""
        python this.py -g transcript_id -s start -e end""")
    parser.add_argument("-f", "--file", dest="file", default=False, type=str, help="region type file")
    parser.add_argument("-t", "--txid", dest="txid", default=False, type=str, help="transcript id")
    parser.add_argument("-s", "--start", dest="start", default=False, type=int, help="start position")
    parser.add_argument("-e", "--end", dest="end", default=False, type=int, help="end position")
    parser.add_argument("-c", "--columns", dest="columns", default='0,1,2', type=str, help="end position")
    parser.add_argument("-o", "--output", dest="output", default='./ensembl.cdna2dna.txt', type=str, help="output file name")
    options = parser.parse_args()
    if not options.file and (not options.txid or not options.start or not options.end):
        parser.print_help()
        exit()
    if options.file:
        options.txid, options.start, options.end = [int(i) for i in options.columns.split(',')]
    return options

def get_location(txid, start, end):
    server = "http://rest.ensembl.org"
    ext = "/map/cdna/" + txid + "/" + start + ".." + end + "?"
    try:
        r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
        if not r.ok:
            r.raise_for_status()
            return None
        decoded = r.json()
        if len(decoded['mappings']) > 1:
            g_starts = []
            g_ends = []
            for i in range(len(decoded['mappings'])):
                # print(decoded['mappings'][i])
                g_starts.append(decoded['mappings'][i]['start'])
                g_ends.append(decoded['mappings'][i]['end'])
            gstart = min(g_starts)
            gend = max(g_ends)
        else:
            gstart = decoded['mappings'][0]['start']
            gend = decoded['mappings'][0]['end']
        
        gchrom = decoded['mappings'][0]['seq_region_name']
        locations = {
            'transcript_id': txid,
            'transcript_start': start,
            'transcript_end': end,
            'genome_chrom': gchrom,
            'genome_start': gstart,
            'genome_end': gend,
            'genome_length': abs(gend - gstart + 1) 
        }
        return locations
    except Exception as e:
        print(e)
        print(f'Check {txid}:{start}-{end}')
        return None
    
    
def check_txid(df: pd.DataFrame, tx_col: int = 0):
    """检查dataframe中的转录本ID,如果包含'.'则分割后取第一部分"""
    return df.iloc[:, tx_col].str.split('.').str[0]
    

def get_target_region(df: pd.DataFrame, tx_col: int = False, start_col: int = False, end_col: int = False) -> pd.DataFrame:
    """给定一个dataframe, 根据指定列获取目标区域的坐标"""
    if start_col and end_col:
        df = df.iloc[:, [tx_col, start_col, end_col]]
    else:
        df = df.iloc[:, [0, 1, 2]]
    return df


def filter_row(df: pd.DataFrame, tx_col: int=0):
    """过滤特定列中没有EN开头的行"""
    return df[df.iloc[:, tx_col].str.startswith('EN')]
    
# 针对dataframe行获取坐标
def get_locations_from_df(df: pd.DataFrame):
    # 遍历每一行, 获取坐标
    idx = 0
    locations = {}
    for index, row in df.iterrows():
        txid = str(row.iloc[0])
        start = str(row.iloc[1])
        end = str(row.iloc[2])
        print(f'{txid}:{start}-{end}')
        loci = get_location(txid, start, end)
        if loci:
            locations[idx] = loci
            idx += 1
    # 字典转换为df
    locations = pd.DataFrame.from_dict(locations, orient='index')
    print(locations.head())
    print('...')
    return locations

if __name__ == '__main__':
    options = get_options()
    if options.file:
        tx_df = pd.read_csv(options.file, sep='\t',header=None)
        tx_region = filter_row(tx_df, options.txid)
        tx_region = get_target_region(tx_region, options.txid, options.start, options.end)
        tx_region.iloc[:, 0] = check_txid(tx_region, options.txid)
        print(f'Load {tx_region.shape[0]} regions from {options.file}')
        locations = get_locations_from_df(tx_region)
        # 合并
        locations.to_csv(options.output, sep='\t', index=False)
    elif options.txid and options.start and options.end:
        if '.' in options.txid:
            options.txid = options.txid.split('.')[0]
        location = get_location(options.txid, options.start, options.end)
        print(f'{location[0]}\t{location[1]}\t{location[2]}')

# gene_id = sys.argv[1]
# start = sys.argv[2]
# end = sys.argv[3]

# server = "http://rest.ensembl.org"
# ext = "/map/cdna/" + gene_id + "/" + start + ".." + end + "?"

# r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})

# if not r.ok:
#     r.raise_for_status()
#     sys.exit()

# decoded = r.json()
# chr_num = decoded['mappings'][0]['seq_region_name']
# start_pos = decoded['mappings'][0]['start']
# end_pos = decoded['mappings'][0]['end']

# print(chr_num + "\t" + str(start_pos) + "\t" + str(end_pos))

