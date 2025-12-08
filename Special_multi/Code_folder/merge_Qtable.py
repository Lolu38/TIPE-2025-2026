import csv

Final_file = "Save_folder/Merged_table/tot_q_table.csv"
Default_start_folder = "Save_folder/worker"

def Merge_Q_table():  # Merge all Q-table from all my worker

    for nbr_worker in range(1,5):
        filename = Default_start_folder + str(nbr_worker) + "/q_table.csv"
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for line in reader:
                with open(Final_file, "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(line)

Merge_Q_table()