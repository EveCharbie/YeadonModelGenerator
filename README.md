# YeadonModelGenerator

This project aims to approximate the anthropometric measurements from [Yeadon's model](https://doi.org/10.1016/0021-9290(90)90370-I) from pictures of the participant (4 pictures for generic model, 5 pictures for acrobatic specific model). The procedure proposed here has an error rate of 7% on the 95 measurements from the Yeadon's model, ... (...%) on the center of mass position and ... (...%) on the inertia properties compared to the manual measurements. 

The suggested pipeline is composed of three steps:
1) Taking the participant pictures using an Android application.
2) Uploading the pictures to a computer.
3) Generating a .txt file of the mesurement compatible with the [Yeadon library](https://yeadon.readthedocs.io/en/latest/) and a .bioMod file compatible with [biorbd](https://github.com/pyomeca/biorbd).
The complete pipeline takes approximately 5 min, compared to 45 min using the manual measurements and removes the need for an expert to take the measurements.

 Acrobatic bonus: The twist potential as introduced [here](https://doi.org/10.51224/SRXIV.337) is also measured to help coaches take enlightened decisions regarding twisting strategies.

## How to setup the reference measures
Either tutorial on how to position the chess boards (add printable chessboards to the repo).
Or has to take the 2 reference measurements.... depending on the precision we are able to reach.
This is the chessboard reference (if you don't take this one as the reference, note that the chessboard you have to use should be a 6x6 chessboard):
<p align="center">
    <img
      src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/chessboardRef.png"
    />
</p>

# Taking pictures
## Setup
<p align="center">
    <img
      src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/chessboardx4.jpg"
    />
</p>

To start, you will have to create a square structure measuring 150x150cm, incorporating a chessboard pattern at each corner. Ensure precision in the arrangement, maintaining a distance of 150cm between the centers of adjacent chessboards.
<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/frontSilhouette.jpg" width="150" />
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/sideSilhouette.jpg" width="150"/>
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/rTuckSilhouette.jpg" width="150"/>
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/frontTuckSilhouette.jpg" width="150"/>
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/pikeSilhouette.jpg" width="150"/>
</p>

In terms of spatial parameters. The distance between the individual holding the camera and the wall bearing the chessboard square should measure 350cm. Simultaneously, maintain a distance of 50cm between the person capturing the photos and the designated wall. For enhanced accuracy, position the camera at a height approximately half of the square's width, approximately 75cm.
## Pictures
The position of the person capturing the photos should follow the silhouette in the overlay you don't have to match exactly the silhouete but the position should be maintained.
You will have to take 5 pictures, for every picture the person don't have to be inside the square overlay but all his body should be inside the photo.

## How to install
Once done, proceed to download our application, im2meas, using the Flutter run command in the flutter folder (you will have more information in the flutter folder).This application serves as an overlay for camera functionality, featuring a square interface. It is imperative to position four chessboard squares within the confines of the red square in the overlay. A silhouette guide is provided for reference to facilitate accurate placement of the body.

First, download anaconda [here](https://www.anaconda.com/download).
then, you can create the python environment using the command:
```bash
conda env create -f environment.yml
```

# Getting Started
After taking the 5 pictures, you have to put them in the img folder ("THE_NAME" is the input you entered in the app):
THE_NAME_front_img - THE_NAME_r_tuck_img - THE_NAME_side_img - THE_NAME_tuck_img - THE_NAME_pike_img
```bash
make run
```
it will create a .txt file named: THE_NAME.txt.
This command will also create a folder named THE_NAME_dir where you will have all your modified images to check manually.

If you want you can enter the mass to have better results using the command:
```bash
make run_with_mass mass=MASS
```
To create the .bioMod you can just use the command:

```bash
make biomake name=THE_NAME
```
The .bioMod will appear in the root folder.

To visualize the 3d body model you can use the command:
```bash
make bioviz name=THE_NAME
```
# Optionnal features
## Camera calibration
For more precision, you can perform camera calibration (for more information on the [documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)). Although this process may not guarantee success in all cases, it improves accuracy. To initiate calibration, follow these steps:

Capture at least 10 images of a chessboard from various angles. Exemplary images can be found in the 'tests/pictures/chessboard' directory.

Organize the images and place them in the 'img/chessboard' folder.
After that you can use the following command to start the script witht the camera calibration:
```bash
make run_calibration
```
