# im2meas_camera

im2meas is a Flutter application designed to facilitate body measurement using a live camera overlay. Follow the instructions to accurately capture and analyze body measurements.
This application is desined for Android smartphones.

## Getting Started

First you will have to download flutter:
```bash
 sudo snap install flutter --classic
```
Then before dowloading the app you will have to unlock the developper mode in your android. The method to activate it may differ depending of the model of your device (check how to unlock it on the internet). After that you will be able to find a developper setting where you can activate the developper mode and also allow USB debugging.
After that, you can plug you usb cable and connect it to your device (be sure that you allow file transfert) and to ensure your device is connected to your computer you can run the command and check if your device is connected:
```bash
 flutter devices
```
Then you just have to run the flutter run command in the flutter folder:
```bash
 flutter run
```

# Instructions:

Upon launching the application, users are prompted to grant access to their camera and subsequently input the name of the individual conducting the measurements.
After that, the user are presented with a screen featuring a live camera preview overlaid with a red square and four smaller blue squares. These smaller squares are strategically positioned to align precisely with four designated chessboards on the wall. Additionally, a silhouette guide is overlaid, indicating the positioning for the individual taking the measurements. While exact matching with the silhouette is not mandatory, the exact position of the silhouette is crucial.
Upon capturing the image, users are given the option to either keep or delete the picture, with the additional functionality to reshoot if you choosed to delete.
