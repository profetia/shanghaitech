import Bio
#use BioPython to read the fasta file
from Bio import SeqIO
file_path="dataset/human/GRCh38_latest_genomic.fna"
#read the fasta file
records = list(SeqIO.parse(file_path, "fasta"))
#check the number of sequences
print(len(records))
#check the first sequence
#print(records[0].seq)
#check the first sequence's description
print(records[0].description)
#check the first sequence's length
print(len(records[0].seq))
