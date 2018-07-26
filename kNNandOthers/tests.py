fout=open("allData.csv","a")
# first file:
for line in open("cell0.csv"):
    fout.write(line)
# now the rest:
for num in range(2, 10):
    f = open("cell"+str(num)+".csv")
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()