import csv

fichier1 = "Dossier_enregistrement/full_ray/scores.csv"
fichier2 = "Dossier_enregistrement/stop_herbe/scores.csv"
colonne = "total_score"
taille_tranche = 50

with open(fichier2, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    
    valeurs = []
    ligne_num = 0
    
    for row in reader:
        ligne_num += 1
        valeur = row[colonne]
        if valeur:  # ignorer les cases vides
            valeurs.append(float(valeur))
        
        # Dès qu'on a rempli une tranche, calculer la moyenne
        if ligne_num % taille_tranche == 0:
            moyenne = sum(valeurs) / len(valeurs) if valeurs else 0
            start = ligne_num - taille_tranche + 1
            end = ligne_num
            print(f"Lignes {start} à {end} : moyenne = {moyenne}")
            valeurs = []  # réinitialiser pour la tranche suivante

    # Calculer la moyenne pour la dernière tranche s'il reste des valeurs
    if valeurs:
        start = ligne_num - len(valeurs) + 1
        end = ligne_num
        moyenne = sum(valeurs) / len(valeurs)
        print(f"Lignes {start} à {end} : moyenne = {moyenne}")
