import csv

fichier1 = "Save_folder_phase1/scores.csv"
colonne = "total_score"
taille_tranche = 200
nbr_sup__nulle = 0
nbr_sup__100 = 0
nbr_sup__50 = 0
nbr_sup_0 = 0
nbr_sup_100 = 0
nbr_sup_150 = 0
nbr_sup_200 = 0
nbr_sup_250 = 0
nbr_sup_300 = 0
nbr_sup_350 = 0
nbr_sup_400 = 0

def min_max_tranche(tab):
    return [min(tab), max(tab)]

for fichier in [fichier1]:
    print(f"Traitement du fichier : {fichier}")
    with open(fichier, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        valeurs = []
        ligne_num = 0
        min_val = float('inf')
        max_val = float('-inf')
        
        for row in reader:
            ligne_num += 1
            valeur = row[colonne]
            if valeur:  # ignorer les cases vides
                valeurs.append(float(valeur))
                # Mettre à jour min et max
                if float(valeur) < min_val:
                    min_val = float(valeur)
                if float(valeur) > max_val:
                    max_val = float(valeur)
                
                if float(valeur) <= -100:
                    nbr_sup__nulle += 1
                if float(valeur) > -100 and float(valeur) <= -50:
                    nbr_sup__100 += 1
                if float(valeur) > -50 and float(valeur) <= 0:
                    nbr_sup__50 += 1
                if float(valeur) > 0 and float(valeur) <= 100:
                    nbr_sup_0 += 1
                if float(valeur) > 100 and float(valeur) <= 150:
                    nbr_sup_100 += 1
                if float(valeur) > 150 and float(valeur) <= 200:
                    nbr_sup_150 += 1
                if float(valeur) > 200 and float(valeur) <= 250:
                    nbr_sup_200 += 1
                if float(valeur) > 250 and float(valeur) <= 300:
                    nbr_sup_250 += 1
                if float(valeur) > 300 and float(valeur) <= 350:
                    nbr_sup_300 += 1
                if float(valeur) > 350 and float(valeur) <= 400:
                    nbr_sup_350 += 1
                if float(valeur) > 400:
                    nbr_sup_400 += 1
            
            # Dès qu'on a rempli une tranche, calculer la moyenne
            if ligne_num % taille_tranche == 0:
                moyenne = sum(valeurs) / len(valeurs) if valeurs else 0
                start = ligne_num - taille_tranche + 1
                end = ligne_num
                tranche = []
                for index in range(taille_tranche):
                    tranche.append(valeurs[index])
                min_tranche, max_tranche = min_max_tranche(tranche)
                print(f"Lignes {start} à {end} : moyenne = {moyenne} | Minimum = {min_tranche} | Max = {max_tranche}")
                valeurs = []  # réinitialiser pour la tranche suivante

        # Calculer la moyenne pour la dernière tranche s'il reste des valeurs
        if valeurs:
            start = ligne_num - len(valeurs) + 1
            end = ligne_num
            moyenne = sum(valeurs) / len(valeurs)
            tranche = []
            for index in range(len(valeurs)):
                tranche.append(valeurs[index])
            min_tranche, max_tranche = min_max_tranche(tranche)
            print(f"Lignes {start} à {end} : moyenne = {moyenne} | Minimum = {min_tranche} | Max = {max_tranche}")
 
        

    print(f"Valeur minimale : {min_val}")
    print(f"Valeur maximale : {max_val}")
    print(f"Nombre de scores inférieurs à -100 : {nbr_sup__nulle}")
    print(f"Nombre de scores entre -100 et -50 : {nbr_sup__100}")
    print(f"Nombre de scores entre -50 et 0 : {nbr_sup__50}")
    print(f"Nombre de scores entre 0 et 100 : {nbr_sup_0}")
    print(f"Nombre de scores entre 150 et 200 : {nbr_sup_150}")
    print(f"Nombre de scores entre 200 et 250 : {nbr_sup_200}")
    print(f"Nombre de scores entre 250 et 300 : {nbr_sup_250}")
    print(f"Nombre de scores entre 300 et 350 : {nbr_sup_300}")
    print(f"Nombre de scores entre 350 et 400 : {nbr_sup_350}")
    print(f"Nombre de scores supérieur à 400 : {nbr_sup_400}")