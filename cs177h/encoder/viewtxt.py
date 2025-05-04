#open txt
with open("dataset/human/GRCh38_latest_genomic_5mer.txt", "r") as output:
    kmer = output.read()
    print(kmer[:300])
    output.close()