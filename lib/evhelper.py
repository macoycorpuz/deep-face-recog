import csv
import pickle

#Save embedded vector to pkl function
def save_embedded_vectors_to_pkl(filename, embedded_vectors):
    output = open(filename, 'wb')
    pickle.dump(embedded, output)
    output.close()
    print("Embedded Vectors successfully saved to " + filename)

#Save embedded vector to csv function
def save_embedded_vectors_to_csv(filename, embedded_vectors):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for img_vector in embedded_vectors:
            spamwriter.writerow(img_vector)
        print("Embedded Vectors successfully saved to " + filename)
