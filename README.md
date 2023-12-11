# YeadonModelGenerator

The primary objective of this project is to generate a three-dimensional human body model employing the Yeadon model, utilizing only four picturess.

## How to install
First, download anaconda it will easier.
You can create the environment using the command:
```bash
conda env create -f environment.yml
```

# Taking pictures
## Setup
<p align="center">
    <img
      src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/chessboardx4.jpg"
    />
</p>

To commence you will have to create a square structure measuring 150x150cm, incorporating a chessboard pattern at each corner. Ensure precision in the arrangement, maintaining a distance of 150cm between the centers of adjacent chessboards.
<p style="display: flex;align-items: center;justify-content: center;">
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/front_silhouette.jpg" width="200" />
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/side_silhouette.jpg" width="200"/>
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/r_pike_silhouette.jpg" width="200"/>
  <img src="https://github.com/Hakuou123/YeadonModelGenerator/blob/main/tests/pictures/front_pike_silhouette.jpg" width="200"/>
</p>
Once done, proceed to download our application, im2meas, utilizing the Flutter run command in the flutter folder. This application serves as an overlay for camera functionality, featuring a square interface. It is imperative to position four chessboard squares within the confines of the red square in the overlay. A silhouette guide is provided for reference to facilitate accurate placement of the body.

In terms of spatial parameters. The distance between the individual holding the camera and the wall bearing the chessboard square should measure 355cm. Simultaneously, maintain a distance of 50cm between the person capturing the photos and the designated wall. For enhanced accuracy, position the camera at a height approximately half of the square's width, approximately 75cm.
## Pictures
The position of the person capturing the photos should follow the silhouette in the overlay yu don't have to match exactly the silhouete but the position should be maintained.
You will have to take 4 pictures, for every picture the person dont have to be inside the square overlay but all his body should be inside the photo.

# Getting Started
After taking the 4 pictures, you have to put them in the img file, it should be in order:
THE_NAME_front_img - THE_NAME_r_tuck_img - THE_NAME_side_img - THE_NAME_tuck_img
```bash
make run
```
it will create a .txt file named: THE_NAME.txt ("THE_NAME" is the input you entered in the app).
To create the .bioMod you can just use the command:

```bash
make biomake name=THE_NAME
```

To visualize the 3d body model you can use the command:
```bash
make bioviz name=THE_NAME
```
