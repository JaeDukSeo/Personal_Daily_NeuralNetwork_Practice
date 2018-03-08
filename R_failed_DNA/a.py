# Read the file and get the DNA string
# file = open('sample_dna.txt', 'r')
# dna = file.read()
import random
import sys

# DNA codon table
protein = {"TTT" : "F", "CTT" : "L", "ATT" : "I", "GTT" : "V",
           "TTC" : "F", "CTC" : "L", "ATC" : "I", "GTC" : "V",
           "TTA" : "L", "CTA" : "L", "ATA" : "I", "GTA" : "V",
           "TTG" : "L", "CTG" : "L", "ATG" : "M", "GTG" : "V",
           "TCT" : "S", "CCT" : "P", "ACT" : "T", "GCT" : "A",
           "TCC" : "S", "CCC" : "P", "ACC" : "T", "GCC" : "A",
           "TCA" : "S", "CCA" : "P", "ACA" : "T", "GCA" : "A",
           "TCG" : "S", "CCG" : "P", "ACG" : "T", "GCG" : "A",
           "TAT" : "Y", "CAT" : "H", "AAT" : "N", "GAT" : "D",
           "TAC" : "Y", "CAC" : "H", "AAC" : "N", "GAC" : "D",
            "CAA" : "Q", "AAA" : "K", "GAA" : "E",
            "CAG" : "Q", "AAG" : "K", "GAG" : "E",
           "TGT" : "C", "CGT" : "R", "AGT" : "S", "GGT" : "G",
           "TGC" : "C", "CGC" : "R", "AGC" : "S", "GGC" : "G",
            "CGA" : "R", "AGA" : "R", "GGA" : "G",
           "TGG" : "W", "CGG" : "R", "AGG" : "R", "GGG" : "G" 
           }



dna = "ATGGAAGTATTTAAAGCGCCACCTATTGGGATATAA"
num_training_data = 5
RNA = ['A','G','T','C']
Terminal_Sig = ['TAA','TAG','TGA']

for _ in range(num_train):
    
    current_data = ""
    protein_sequence = ""
    protein_test = ""
    
    for _ in range(30):
        rna,protein_data = random.choice(list(protein.items()))
        current_data = current_data + rna
        protein_sequence = protein_sequence + protein_data

    current_data = current_data + 'TGA'

    # Generate protein sequence
    for i in range(0, len(current_data)-(3+len(current_data)%3), 3):
        if protein[current_data[i:i+3]] == "STOP" :
            break
        protein_test += protein[current_data[i:i+3]]

    print("Made Protein: ",current_data )
    print("Made Protein: ",protein_sequence )
    print("test Protein: ",protein_test )
    
    



sys.exit()

dna = "ATGGAAGTATTTAAAGCGCCACCTATTGGGATATAA"
print("DNA Sequence: ", dna)
print(len(dna))


protein_sequence = ""

# Generate protein sequence
for i in range(0, len(dna)-(3+len(dna)%3), 3):
    if protein[dna[i:i+3]] == "STOP" :
        break
    protein_sequence += protein[dna[i:i+3]]

# Print the protein sequence
print("Protein Sequence: ", protein_sequence)
print(len(protein_sequence))

# End of program