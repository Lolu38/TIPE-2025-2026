import cv2
import csv

track = 4
# Charger ton image du circuit
img = cv2.imread(f"./Save_folder/tracks/track{track}/full_track_seed{track}.png")
clone = img.copy()
checkpoints = []

def click_event(event, x, y, flags, param):
    global checkpoints
    if event == cv2.EVENT_LBUTTONDOWN:
        checkpoints.append((x, y))
        # afficher un cercle rouge sur le point cliqué
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Circuit", img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # retirer le dernier point avec clic droit
        if checkpoints:
            checkpoints.pop()
            img[:] = clone.copy()
            for (cx, cy) in checkpoints:
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow("Circuit", img)

cv2.imshow("Circuit", img)
cv2.setMouseCallback("Circuit", click_event)
print("🔹 Clique gauche pour ajouter un checkpoint, droit pour annuler, q pour quitter.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Sauvegarder les checkpoints
with open(f"./Save_folder/tracks/track{track}/checkpoints.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for point in checkpoints:
        writer.writerow(point)

print(f"✅ {len(checkpoints)} checkpoints sauvegardés dans checkpoints.csv")
