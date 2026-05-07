import os
import matplotlib.pyplot as plt
import cv2

# ---------------------------
# Set dataset paths
# ---------------------------
real_path = "Dataset/Train/Real"
fake_path = "Dataset/Train/Fake"

real_images = os.listdir(real_path)
fake_images = os.listdir(fake_path)

# Use a subset for faster testing (remove later for full dataset)
#real_images = real_images[:500]
#fake_images = fake_images[:500]

# ---------------------------
# Task 1: Count images
# ---------------------------
num_real = len(real_images)
num_fake = len(fake_images)

print("Real:", num_real)
print("Fake:", num_fake)

# Plot 1: Real vs Fake
labels = ["Real", "Fake"]
counts = [num_real, num_fake]

plt.bar(labels, counts)
plt.title("Number of Real vs Fake Images")
plt.xlabel("Category")
plt.ylabel("Count")
plt.savefig("real_vs_fake.png")  # overwrite each run
plt.show()

# ---------------------------
# Face detection function (replacement for get_faces)
# ---------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def get_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# ---------------------------
# Task 2: Count images without detectable faces
# ---------------------------
no_face_count = 0
total = 0

# Process real images
for img_name in real_images:
    img_path = os.path.join(real_path, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    faces = get_faces(image)
    total += 1

    if len(faces) == 0:
        no_face_count += 1

# Process fake images
for img_name in fake_images:
    img_path = os.path.join(fake_path, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    faces = get_faces(image)
    total += 1

    if len(faces) == 0:
        no_face_count += 1

print("Total images:", total)
print("Images without faces:", no_face_count)

# ---------------------------
# Plot 2: Face detection distribution
# ---------------------------
labels = ["Has Face", "No Face"]
sizes = [total - no_face_count, no_face_count]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Face Detection Distribution")
plt.savefig("face_detection.png")  # overwrite each run
plt.show()
