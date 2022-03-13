from sqlite3 import connect

from pandas import read_csv

cnxn = connect("stats_rethink.db")

#howell = read_csv("Howell1.csv", delimiter=";")
#print(howell)
#howell.to_sql("howell1", cnxn, if_exists="fail", index=False)

#cherry_blossom = read_csv("cherry_blossoms.csv")
#cherry_blossom.to_sql("kyoto", cnxn, if_exists="fail", index=False)

# Not quite correct.
#files = glob("*.csv")
#for file in files:
#    df = read_csv(file, delimiter=";")
#    table_name = file.rsplit(".")[0]
#    df.to_sql(table_name, cnxn, if_exists="replace", index=False)
