
#  Implementation of an Image Encryption algorithm based on a chaotic system and Brownian motion

This project is an implmentation of an image encryption algorithm based on a chaotic system proposed by Zhao, Meng, Zhang and Yang.

The project can take in a greyscale image and encrypt it on the basis of the given algorithm, which has been shown to be robust against attacks.

This is a full implmentation of the existing algorithm, which I've also adapted to work on color images.

At the moment, I have only optimized the greyscale version and made it more interactive. The color version will be updated in time.

## Files

There are four main files for this project.

* `imageEncrypter.py`: used to encrypt the (greyscale) image.
* `imageDecrypter.py`: used to decrypt the (greyscale) image.
* `colorImageEncrypter.py`: used to encrypt the (color) image.
* `colorImageDecrypter.py`: used to decrypt the (color) image.

There are sample test images that were used in testing the algorithm, in the `TestImages` directory. If you would just like to view the results of encrypting and decrypting, please see the `FinalImages` directory.

In the `FinalImages` directory, the file format is as follows:

* `Grey<name>.png`: the input test image
* `GreyEncrypted<name>.png`: the encrypted version of the image with the same name, with default parameters
* `GreyDecrypted<name>.png`

## Running the files

To use the encryption file, all that is necessary is to run `imageEncrypter.py`, and provide the relative path to a P2 `.ppm` image file (or P3 mode `.ppm` in the case of using color images) when prompted.

The encrypted image will be stored in the directory `.\encryptedImages\*.ppm`, where "*" will be replaced with the name of the file.

To run the decryption file, provide the same path to file when prompted, and the decrypted image will be in the directory `.\decryptedImages\*.ppm`, using the same naming convention as above.